"""Utility functions for ALFA based data selection strategy."""




from dataclasses import dataclass
from functools import lru_cache
from typing import Any


import lancedb
import numpy as np
import numpy.typing as npt


from autonomy.perception.datasets.active_learning.alfa_curate.data_types import RowIdComponents
from autonomy.perception.datasets.features.cosmos.infer import Cosmos
from autonomy.perception.datasets.log_slices.logapps_metadata.data_model import LogappsMetadata
from kits.ml.datasets.identifiers import Identifiers
from kits.scalex.dataset.instances.lance_dataset import read_database_path
from platforms.lakefs.client import LakeFS




@dataclass
class SearchResult:
   """Result from text-to-video similarity search.


   Simililar to EmbeddedVideo with distance added (autonomy/perception/datasets/features/cosmos_index/generate_cosmos_index.py).
   """


   identifiers: Identifiers
   row_id: str
   logapps_metadata: LogappsMetadata | None
   sensor_name: str
   start_ns: np.int64
   end_ns: np.int64
   image_paths: list[str]
   embedding: npt.NDArray[np.float32]
   distance: float


   @property
   def similarity(self) -> float:
       """Convert l2 distance 0~sqrt(2) to similarity score 0~1."""
       return l2_distance_to_similarity(self.distance)




@lru_cache(maxsize=1)
def load_table(repo: str, branch: str, table_name: str) -> Any:
   """Loads the lancedb table."""


   lakefs = LakeFS()
   db_path = read_database_path(repo, branch, lakefs)
   db = lancedb.connect(db_path)
   return db.open_table(table_name)




@lru_cache(maxsize=2)
def load_model(model_size: str) -> Cosmos:
   """Load and cache Cosmos model."""
   return Cosmos(model_size=model_size, load_model_from_lakefs=False)




def text_to_embedding(query: str, model_size: str) -> npt.NDArray[np.float32]:
   """Generate text embedding."""
   model = load_model(model_size)
   return model.text_embedding(query)




def run_text_query(table: Any, query: str, top_k: int, model_size: str) -> list[SearchResult]:
   """Search for text query in Cosmos embeddings using LanceDB."""
   vec = text_to_embedding(query, model_size)
   results = table.search(vec, vector_column_name="embedding").limit(top_k).to_list()
   return [
       SearchResult(
           identifiers=r["identifiers"],
           row_id=r["row_id"],
           logapps_metadata=r["logapps_metadata"],
           sensor_name=r["sensor_name"],
           start_ns=r["start_ns"],
           end_ns=r["end_ns"],
           image_paths=r["image_paths"],
           embedding=r["embedding"],
           distance=r["_distance"],
       )
       for r in results
   ]




def l2_distance_to_similarity(distance: float) -> float:
   """Convert LanceDB l2 distance to similarity in [0, 1]."""
   return max(0.0, 1.0 - (float(distance) / 2.0))




def parse_row_id(row_id: str) -> RowIdComponents:
   """Process row_id to get slice_id, start_ns, end_ns, and camera_name."""
   if "_segment_" not in row_id:
       return RowIdComponents(row_id, "", "", "")


   base_row_id, remainder = row_id.split("_segment_", 1)


   # Split remainder - only split on first 2 underscores
   parts = remainder.split("_", 2)
   if len(parts) != 3:
       return RowIdComponents(base_row_id, "", "", "")


   return RowIdComponents(base_row_id, parts[0], parts[1], parts[2])




def deduplicate_by_base_slice(results: list[SearchResult]) -> list[SearchResult]:
   """Keep only the best (highest similarity) hit per base slice id."""
   dedup_by_base: dict[str, SearchResult] = {}


   for result in results:
       if not result.row_id:
           continue
       base_slice_id = parse_row_id(str(result.row_id))[0]
       similarity = result.similarity


       if base_slice_id not in dedup_by_base or similarity > dedup_by_base[base_slice_id].similarity:
           dedup_by_base[base_slice_id] = result


   deduped = list(dedup_by_base.values())
   deduped.sort(key=lambda x: x.similarity, reverse=True)
   return deduped
