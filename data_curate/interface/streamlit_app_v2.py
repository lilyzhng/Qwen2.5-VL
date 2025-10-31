"""Search for text-to-video, video-to-video over Cosmos embeddings (LanceDB index).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Final, Literal, NamedTuple, Optional, cast

import fsspec
import grpc
import imageio
import lancedb
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st

from argo_log_apps_presigned_url_service_sdk.ai.argo.log_apps.presignedurlservice.v1 import (
    service_pb2 as presigned_url_service_pb2,
)
from argo_log_apps_presigned_url_service_sdk.ai.argo.log_apps.presignedurlservice.v1 import (
    service_pb2_grpc as presigned_url_service_grpc,
)
from autonomy.perception.datasets.features.cosmos.config import COSMOS_MODEL_SIZE
from autonomy.perception.datasets.features.cosmos.infer import Cosmos
from kits.scalex.dataset.instances.lance_dataset import read_database_path
from kits.scalex.hpc.zip_uri_reader_utils import parse_uri
from lat_ojo import ojo
from liam_common_py.interceptors.client.defaults import get_default_auth_intercepted_channel
from liam_common_py.providers.defaults import get_default_liam_provider
from liam_common_py.utils.grpc_utils import get_grpc_ssl_credentials
from log_apps.python.public.libs.api_client.event import query_event_details
from platforms.data.streamlat import bq
from platforms.data.streamlat.visualizing import who_are_you
from platforms.data.streamlat.visualizing.event_viewer import LatitubeConfig, show_iframe
from platforms.lakefs.client import LakeFS


def parse_row_id(row_id: str) -> tuple[str, str, str, str]:
    """Process row_id to get slice_id, start_ns, end_ns, and camera_name."""
    if "_segment_" not in row_id:
        return row_id, "", "", ""

    base_row_id, remainder = row_id.split("_segment_", 1)

    # Split remainder - only split on first 2 underscores
    parts = remainder.split("_", 2)
    start_ns = parts[0]
    end_ns = parts[1]
    camera_name = parts[2]

    return base_row_id, start_ns, end_ns, camera_name


def get_similarity_score(video: dict[str, Any]) -> float:
    """Convert LanceDB distance to similarity score."""
    distance = video["_distance"]
    return max(0.0, 1.0 - (float(distance) / 2.0))


def deduplicate_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate search results by base video ID, keeping the best scoring clip from each video.

    Args:
        results: List of search result dictionaries with row_id and _distance

    Returns:
        Deduplicated list with one representative clip per base video
    """
    if not results:
        return results

    # Group results by base video ID
    video_groups: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        row_id = result.get("row_id", "")
        base_video_id, start_ns, end_ns, camera_name = parse_row_id(row_id)

        # Use base_video_id + camera_name as the grouping key to handle multiple cameras
        group_key = f"{base_video_id}_{camera_name}" if camera_name else base_video_id

        if group_key not in video_groups:
            video_groups[group_key] = []
        video_groups[group_key].append(result)

    # Select the best representative from each group (lowest distance = highest similarity)
    deduplicated_results = []
    for group_key, group_results in video_groups.items():
        # Sort by distance (lower is better) and take the first one
        best_result = min(group_results, key=lambda x: x.get("_distance", float("inf")))
        deduplicated_results.append(best_result)

    # Sort deduplicated results by original ranking (distance)
    deduplicated_results.sort(key=lambda x: x.get("_distance", float("inf")))

    return deduplicated_results


@st.cache_resource
def get_lakefs() -> LakeFS:
    """Gets lakefs client instance."""
    return LakeFS()


@st.cache_resource
def load_model(model_size: str) -> Cosmos:
    """Loads the Cosmos model."""
    return Cosmos(model_size=model_size, load_model_from_lakefs=False)


@st.cache_resource
def load_table(branch: str) -> Any:
    """Loads the lancedb table."""
    lakefs = get_lakefs()
    db_path = read_database_path(REPO, branch, lakefs)
    db = lancedb.connect(db_path)
    return db.open_table(TABLE)


@st.cache_resource
def get_presigned_url_service() -> Any:
    """Gets the presigned URL service gRPC client."""
    endpoint = "presigned-url-service.log-apps.bluel3.tools:443"
    grpc_credentials = get_grpc_ssl_credentials()
    base_channel = grpc.secure_channel(endpoint, grpc_credentials)
    provider = get_default_liam_provider(host_or_channel=endpoint)
    channel = get_default_auth_intercepted_channel(base_channel, provider=provider)
    return presigned_url_service_grpc.PresignedUrlServiceStub(channel)


@st.cache_data()
def get_video_url_from_road_event_uuid(road_event_uuid: str) -> Optional[str]:
    """Gets the video URL from the road event UUID."""
    results = query_event_details(road_event_uuid)
    if "_source" not in results or "video_requests" not in results["_source"]:
        return None

    video_request = results["_source"]["video_requests"]
    preview_link = None
    full_link: str | None = None

    for request in video_request:
        if request["video_type"] != "POTATO":
            continue

        preview_link = request.get("preview_path")
        full_link = request.get("video_s3_path")

    if full_link is not None:
        return full_link
    return preview_link


def get_presigned_url_for_hpc_file(hpc_path: str) -> str:
    """Gets the pre-signed URL of the HPC file."""
    presigned_url_service = get_presigned_url_service()
    request = presigned_url_service_pb2.GetPresignedUrlRequest()
    request.file_path = hpc_path
    response = presigned_url_service.GetPresignedUrl(request)
    return str(response.presigned_url)


def text_to_embedding(query: str, model_size: str) -> npt.NDArray[np.float32]:
    """Converts text query to embedding."""
    model = load_model(model_size)
    return model.text_embedding(query)


def run_text_query(
    branch: str, query: str, k: int, model_size: str
) -> list[dict[str, Any]]:
    """Runs a K-NN text-to-video similarity search."""
    table = load_table(branch)
    vec = text_to_embedding(query, model_size)
    results = table.search(vec, vector_column_name="embedding").limit(k).to_list()
    return cast(list[dict[str, Any]], results)


def find_slice_in_table(table: Any, slice_id: str) -> Optional[dict[str, Any]]:
    """Find a slice in the table by slice_id and return its row data."""
    try:
        all_results = table.to_pandas()

        if "row_id" in all_results.columns:
            matching_rows = all_results[all_results["row_id"] == slice_id]
            if not matching_rows.empty:
                return cast(dict[str, Any], matching_rows.iloc[0].to_dict())

    except Exception as e:
        logging.warning("Error finding video in table", exc_info=e)

    return None


def run_video_query(
    branch: str, slice_id: str, k: int
) -> tuple[list[dict[str, Any]], bool]:
    """Runs a K-NN video-to-video similarity search using precomputed embeddings only."""
    table = load_table(branch)

    video_row = find_slice_in_table(table, slice_id)

    if video_row and "embedding" in video_row:
        # Use precomputed embedding with self-avoidance
        logging.info("Using precomputed embedding for: %s", slice_id)
        query_embedding = video_row["embedding"]

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        results = (
            table.search(query_embedding, vector_column_name="embedding")
            .limit(k + 1)
            .to_list()
        )

        input_row_id = video_row.get("row_id", "")
        filtered_results = [r for r in results if r.get("row_id", "") != input_row_id][
            :k
        ]

        return cast(list[dict[str, Any]], filtered_results), True
    else:
        logging.info("No precomputed embedding found for: %s", slice_id)
        return [], False


def metadata_summary(row: dict[str, Any]) -> str:
    """Gets metadata summary for a slice."""
    identifiers = row.get("identifiers", {})
    slice_id = (
        identifiers.get("slice_id", "<unknown>")
        if isinstance(identifiers, dict)
        else "<unknown>"
    )
    slice_id, start_ns, end_ns, camera_name = parse_row_id(slice_id)
    tags = None
    metadata = row.get("logapps_metadata")
    if isinstance(metadata, dict):
        tags = metadata.get("tags")
    tag_str = (
        ", ".join(tags)
        if isinstance(tags, list)
        else str(tags)
        if tags
        else "<no tags>"
    )
    return (
        "🔗 Slice: https://lats.bluel3.tools/log-apps/slices/"
        f"{slice_id} | Sensor: {row.get('sensor_name')} | Timespan: {start_ns, end_ns} | Tags: {tag_str}"
    )


@st.cache_data
def get_image(image_uri: str) -> npt.NDArray[np.uint8] | None:
    """Retrieves an image from the provided URI.


    The URI can be a local, s3, or zip URI. This function attempts to load the image and returns it.
    The function depends on valid AWS credentials for data on S3 and on lustre being mounted for HPC data.


    Args:
        image_uri: A string representing the image URI.


    Returns:
        NDArray or None: The loaded image as a numpy array if successful, otherwise None.
    """
    if not image_uri.startswith("/s/latai"):
        try:
            with fsspec.open(image_uri, "rb") as fh:
                return imageio.v3.imread(fh)
        except Exception:
            pass

    zip_path, file_path = parse_uri(image_uri)
    path = f"zip://{file_path}::{zip_path}"

    try:
        with fsspec.open(path, "rb") as fh:
            return imageio.v3.imread(fh)
    except Exception as e:
        st.write(e)
        return None


def get_media_html_for_result(
    row: dict[str, Any], viz_option: str = "Video Only"
) -> str:
    """Extract media HTML for a result row."""
    metadata = row.get("logapps_metadata") or {}
    event_uuid = metadata.get("road_event_uuid")
    hpc_video_link = (
        get_video_url_from_road_event_uuid(event_uuid) if event_uuid else None
    )
    signed_path: str | None = None

    if viz_option == "Image Only":
        image_paths: list[str] = row.get("image_paths", [])

        if image_paths:
            first_image_path = image_paths[0]
            img_array = get_image(first_image_path)
            if img_array is not None:
                buf = BytesIO()
                imageio.v3.imwrite(buf, img_array, format="png")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f'<img src="data:image/png;base64,{b64}" alt="frame" style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;" />'

        return '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #888; font-size: 2rem; background: #f0f0f0; border-radius: 8px;">📷</div>'

    elif viz_option == "Video Only":
        if hpc_video_link:
            try:
                signed_path = get_presigned_url_for_hpc_file(hpc_video_link)
                if signed_path:
                    return f'<video src="{signed_path}" controls preload="metadata" playsinline style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;"></video>'
            except Exception:
                logging.warning(
                    "Failed to get pre-signed URLs for video on HPC.",
                    exc_info=True,
                )

        return '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #888; font-size: 2rem; background: #f0f0f0; border-radius: 8px;">🎬</div>'

    return '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #888; font-size: 2rem; background: #f0f0f0; border-radius: 8px;">❓</div>'


def display_search_results_grid(
    results: list[dict[str, Any]],
    search_type: str,
    search_info: str,
    top_k: int = 2,
    similarity_threshold: float = 0.0,
    viz_option: str = "Video Only",
    deduplicate: bool = True,
) -> None:
    """Display search results in a dynamic grid layout, filtered by similarity threshold."""

    st.markdown(
        """
   <div style="text-align: center; margin-bottom: 1.5rem;">
       <h3 style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">Top-K Results</h3>
   </div>
   """,
        unsafe_allow_html=True,
    )

    input_video_data = None
    if search_type == "video":
        input_slice_id = search_info.replace('Video: "', "").replace('"', "")
        table = load_table(BRANCH)
        input_video_data = find_slice_in_table(table, input_slice_id)

    original_count = len(results)
    filtered_results = [
        r for r in results if get_similarity_score(r) >= similarity_threshold
    ]

    # Apply deduplication for visualization if enabled
    if deduplicate:
        deduplicated_results = deduplicate_search_results(filtered_results)
        deduplicated_count = len(deduplicated_results)
    else:
        deduplicated_results = filtered_results
        deduplicated_count = len(filtered_results)

    # Show filtering and deduplication info
    info_messages = []
    if original_count > len(filtered_results) and similarity_threshold > 0.0:
        filtered_count = original_count - len(filtered_results)
        info_messages.append(
            f"🔍 Filtered out {filtered_count} results below similarity threshold ({similarity_threshold:.2f})"
        )

    if deduplicate and len(filtered_results) > deduplicated_count:
        duplicates_removed = len(filtered_results) - deduplicated_count
        info_messages.append(
            f"🎬 Deduplicated {duplicates_removed} overlapping segments from the same videos"
        )

    if info_messages:
        results_text = "unique videos" if deduplicate else "results"
        combined_message = (
            " | ".join(info_messages)
            + f" | Showing {deduplicated_count} {results_text} of {original_count} total results."
        )
        st.info(combined_message)

    # Use processed results for display
    filtered_results = deduplicated_results

    total_results = len(filtered_results)
    num_results_to_show = min(top_k, total_results)

    input_cols = st.columns([1, 2, 1])

    with input_cols[1]:
        if search_type == "text":
            query_text = search_info.replace('Text: "', "").replace('"', "")
            st.markdown(
                f"""
               <div style="position: relative; margin-bottom: 0.5rem;">
                   <div style="width: 100%; aspect-ratio: 16/9; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                       border-radius: 8px; border: 3px solid #f59e0b; display: flex; align-items: center; justify-content: center; padding: 20px;">
                       <div style="color: white; text-align: center;">
                           <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
                           <div style="font-size: 1.5rem; font-weight: 500; word-wrap: break-word; line-height: 1.3;">
                               "{query_text}"
                           </div>
                       </div>
                   </div>
                   <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
                       background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px;
                       font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                       INPUT
                   </div>
               </div>
               """,
                unsafe_allow_html=True,
            )

            # Text query info
            st.markdown(
                """
           <div style="text-align: center; margin-top: 0.0rem;">
               <div style="font-size: 0.85rem; color: #1e293b;">
                   <span style="font-weight: 600; color: #f59e0b;">Text Query</span>
               </div>
           </div>
           """,
                unsafe_allow_html=True,
            )

        elif search_type == "video":
            input_slice_id = search_info.replace('Video: "', "").replace('"', "")
            input_slice_id, start_ns, end_ns, camera_name = parse_row_id(input_slice_id)

            if input_video_data:
                input_media_html = get_media_html_for_result(
                    input_video_data, viz_option
                )
                st.markdown(
                    f"""
               <div style="position: relative; margin-bottom: 0.5rem;">
                   <div style="width: 100%; aspect-ratio: 16/9; border-radius: 8px; border: 3px solid #f59e0b; overflow: hidden;">
                       {input_media_html}
                   </div>
                   <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
                        background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px;
                        font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                       INPUT
                   </div>
               </div>
               """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
               <div style="position: relative; margin-bottom: 0.5rem;">
                   <div style="width: 100%; aspect-ratio: 16/9; background: #f59e0b; border-radius: 8px; border: 3px solid #f59e0b;
                        display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                       🎬
                   </div>
                   <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
                        background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px;
                        font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                       INPUT
                   </div>
               </div>
               """,
                    unsafe_allow_html=True,
                )

            # Input video info
            st.markdown(
                f"""
           <div style="text-align: center; margin-top: 0.0rem;">
               <div style="font-size: 0.85rem; color: #1e293b;">
                   <a href="https://lats.bluel3.tools/log-apps/slices/{input_slice_id}" target="_blank"
                       style="font-weight: 600; color: #f59e0b; text-decoration: none;">
                       🔗 Slice: {input_slice_id[:8]}
                   </a>
               </div>
           </div>
           """,
                unsafe_allow_html=True,
            )

    if num_results_to_show > 0:
        num_result_rows = (num_results_to_show + 1) // 2

        for row_idx in range(num_result_rows):
            result_cols = st.columns(2)

            for col_idx in range(2):
                result_idx = row_idx * 2 + col_idx

                if result_idx < num_results_to_show:
                    video = filtered_results[result_idx]
                    with result_cols[col_idx]:
                        slice_id = video.get("row_id", f"Result {result_idx + 1}")
                        slice_id, start_ns, end_ns, camera_name = parse_row_id(slice_id)
                        similarity_score = get_similarity_score(video)
                        media_html = get_media_html_for_result(video, viz_option)

                        st.markdown(
                            f"""
                       <div style="position: relative; margin-bottom: 0.5rem;">
                           <div style="width: 100%; aspect-ratio: 16/9; border-radius: 8px; border: 2px solid #e2e8f0; overflow: hidden;">
                               {media_html}
                           </div>
                           <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
                                background: rgba(0,0,0,0.7); color: white; padding: 4px 8px; border-radius: 4px;
                                font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                               Rank {result_idx + 1}
                           </div>
                       </div>
                       """,
                            unsafe_allow_html=True,
                        )

                        sensor = video.get("sensor_name") or "?"

                        st.markdown(
                            f"""
                       <div style="text-align: center; margin-top: 0.0rem;">
                           <div style="font-size: 0.85rem; color: #1e293b;">
                               <a href="https://lats.bluel3.tools/log-apps/slices/{slice_id}" target="_blank"
                                  style="font-weight: 600; color: #6366f1; text-decoration: none;">
                                   🔗 Slice: {slice_id[:8]}
                               </a>
                               <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">
                                   Score: {similarity_score:.3f}
                               </span>
                           </div>
                           <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.0rem;">
                               Embedding Sensor: {sensor}
                           </div>
                       </div>
                       """,
                            unsafe_allow_html=True,
                        )


def display_database_stats(
    num_slices: int, embedding_dim: Any, sensor_names: list[str]
) -> None:
    """Render database statistics in the sidebar."""

    sensor_names_display = ", ".join(sensor_names) if sensor_names else "<unknown>"

    st.markdown(
        '<div class="section-title">📊 Database Stats</div>', unsafe_allow_html=True
    )

    st.markdown(
        f"""
   <div class="stat-card" style="margin-bottom: 0.5rem;">
       <span style="color: #64748b; font-size: 0.9rem;">Total Slices:</span>
       <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">{num_slices:,}</span>
   </div>
   """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
   <div class="stat-card" style="margin-bottom: 0.5rem;">
       <span style="color: #64748b; font-size: 0.9rem;">Model:</span>
       <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">{MODEL_SIZE}</span>
   </div>
   """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
   <div class="stat-card" style="margin-bottom: 0.5rem;">
       <span style="color: #64748b; font-size: 0.9rem;">Embedding Dim:</span>
       <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">{embedding_dim}</span>
   </div>
   """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
   <div class="stat-card" style="margin-bottom: 0.5rem;">
       <span style="color: #64748b; font-size: 0.9rem;">Sensors:</span>
       <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">{sensor_names_display}</span>
   </div>
   """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
   <div class="stat-card" style="margin-bottom: 0.5rem;">
       <span style="color: #64748b; font-size: 0.9rem;">Backend:</span>
       <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">LanceDB</span>
   </div>
   """,
        unsafe_allow_html=True,
    )


def load_custom_css() -> None:
    """Load custom CSS styling."""
    st.markdown(
        """
   <style>
       /* Import Inter font */
       @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');


       /* Global styling */
       .main {
           font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       }


       /* Header styling */
       .main-header {
           background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
           color: white;
           padding: 2rem;
           border-radius: 20px;
           text-align: center;
           margin-bottom: 2rem;
           position: relative;
           overflow: hidden;
           box-shadow: 0 25px 50px rgba(99, 102, 241, 0.25);
       }


       .main-header::before {
           content: '';
           position: absolute;
           top: 0;
           left: 0;
           right: 0;
           bottom: 0;
           background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.05"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
           pointer-events: none;
       }


       .main-header h1 {
           font-size: 3rem;
           margin-bottom: 0.5rem;
           font-weight: 700;
           background: linear-gradient(45deg, #ffffff, #e0e7ff);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           background-clip: text;
           letter-spacing: -0.02em;
           position: relative;
           z-index: 1;
       }


       .main-header p {
           font-size: 1.2rem;
           opacity: 0.9;
           position: relative;
           z-index: 1;
       }


       /* Mobile responsiveness */
       @media (max-width: 768px) {
           .main-header h1 {
               font-size: 2rem;
           }
           .main-header p {
               font-size: 1rem;
           }
       }


       /* Section titles */
       .section-title {
           font-size: 1.4rem;
           color: #1e293b;
           margin-bottom: 1rem;
           font-weight: 700;
           letter-spacing: -0.01em;
           position: relative;
           padding-bottom: 0.75rem;
       }


       .section-title::after {
           content: '';
           position: absolute;
           bottom: 0;
           left: 0;
           width: 50px;
           height: 3px;
           background: linear-gradient(90deg, #6366f1, #8b5cf6);
           border-radius: 2px;
       }


       /* Stats grid */
       .stats-container {
           display: grid;
           grid-template-columns: 1fr 1fr;
           gap: 0.5rem;
           margin: 0.75rem 0;
       }


       .stat-card {
           background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
           padding: 0.75rem;
           border-radius: 12px;
           text-align: left;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06);
           border: 1px solid rgba(226, 232, 240, 0.5);
           transition: transform 0.3s ease;
       }


       .stat-card:hover {
           transform: translateY(-1px);
       }


       .stat-value {
           font-size: 1.5rem;
           font-weight: 800;
           background: linear-gradient(135deg, #6366f1, #8b5cf6);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           background-clip: text;
           margin-bottom: 0.25rem;
           letter-spacing: -0.02em;
       }


       .stat-label {
           font-size: 0.8rem;
           color: #64748b;
           font-weight: 500;
           letter-spacing: 0.02em;
       }


       /* Button styling */
       .stButton > button {
           background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
           color: white !important;
           border: none !important;
           border-radius: 12px !important;
           padding: 0.75rem 1.5rem !important;
           font-weight: 600 !important;
           transition: all 0.3s ease !important;
           box-shadow: 0 4px 8px rgba(99, 102, 241, 0.2) !important;
           width: 100% !important;
       }


       .stButton > button:hover {
           transform: translateY(-2px) !important;
           box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
       }
   </style>
   """,
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class DatabaseInfo:
    """Database metadata and runtime state.


    Attributes:
        num_slices: Total number of data slices available.
        available_slices: List of slice identifiers available for querying.
        model: Name or identifier of the embedding model in use.
        embedding_dim: Dimensionality of the embedding vectors.
        sensor_names: List of sensor names present in the dataset.
        backend: Database backend identifier (e.g. 'duckdb', 'sqlite').
    """

    num_slices: int
    available_slices: list[str]
    model: str
    embedding_dim: str
    sensor_names: list[str]
    backend: str


def load_database_info(branch: str) -> DatabaseInfo:
    """Load database and extract basic information."""
    table = load_table(branch)
    all_data = table.to_pandas()
    num_slices = len(all_data)
    available_slices = (
        list(all_data["row_id"].unique()) if "row_id" in all_data.columns else []
    )

    embedding_dim = "Unknown"
    try:
        if "embedding" in all_data.columns and not all_data.empty:
            first_embedding = (
                all_data["embedding"].dropna().iloc[0]
                if not all_data["embedding"].dropna().empty
                else None
            )
            if first_embedding is not None:
                embedding_dim = (
                    str(len(first_embedding))
                    if hasattr(first_embedding, "__len__")
                    else "unknown"
                )
    except Exception:
        embedding_dim = "Error"

    sensor_names = (
        list(all_data["sensor_name"].unique())
        if "sensor_name" in all_data.columns
        else [""]
    )

    return DatabaseInfo(
        num_slices=num_slices,
        available_slices=available_slices,
        model=MODEL_SIZE,
        embedding_dim=embedding_dim,
        sensor_names=sensor_names,
        backend="LanceDB",
    )


def display_text_search(branch: str, top_k: int) -> None:
    """Handle text search functionality."""
    text_query = st.text_input(
        "Text Query",
        placeholder="pedestrians crossing street",
        label_visibility="collapsed",
    )

    if st.button("🔍 Search by Text", use_container_width=True):
        if text_query.strip():
            with st.spinner("Searching..."):
                try:
                    results = run_text_query(
                        branch, text_query.strip(), top_k, MODEL_SIZE
                    )
                    if results:
                        st.success(f"Found {len(results)} results!")
                        st.session_state["search_results"] = results
                        st.session_state["search_type"] = "text"
                        st.session_state["search_info"] = (
                            f'Text: "{text_query.strip()}"'
                        )
                    else:
                        st.warning("No results returned.")
                except Exception as e:
                    message = f"❌ Search failed: {e}"
                    st.error(message)
        else:
            st.warning("Please enter a search query")


def display_video_search(
    branch: str, available_slice_ids: list[str], top_k: int
) -> None:
    """Handle video search functionality."""
    if available_slice_ids:
        selected_slice_id = st.selectbox(
            "Select Video (Slice ID)",
            available_slice_ids,
            help="Choose a slice from the database to find similar videos",
            label_visibility="collapsed",
        )

        if st.button("🎥 Search by Video", use_container_width=True):
            with st.spinner("Searching for similar videos..."):
                try:
                    results, found_precomputed = run_video_query(
                        branch, selected_slice_id, top_k
                    )
                    if found_precomputed and results:
                        st.success(
                            f"Found {len(results)} similar videos! ⚡ (Used precomputed embedding)"
                        )

                        st.session_state["search_results"] = results
                        st.session_state["search_type"] = "video"
                        st.session_state["search_info"] = (
                            f'Video: "{selected_slice_id}"'
                        )
                    elif not found_precomputed:
                        message = (
                            f"Video {selected_slice_id} not found in the database."
                        )
                        st.warning(message)
                    else:
                        st.warning("No similar videos found.")
                except Exception as e:
                    message = f"❌ Video search failed: {e}"
                    st.error(message)
    else:
        st.info("No videos found in database")


def display_retrieval_results(
    top_k: int, similarity_threshold: float, viz_option: str, deduplicate: bool = True
) -> None:
    """Render the retrieval content area with search results or instructions."""

    if "search_results" in st.session_state and st.session_state["search_results"]:
        display_search_results_grid(
            st.session_state["search_results"],
            st.session_state.get("search_type", "text"),
            st.session_state.get("search_info", ""),
            top_k,
            similarity_threshold,
            viz_option,
            deduplicate,
        )

        with st.expander("📊 Download Results Table"):
            filtered_table_results = [
                r
                for r in st.session_state["search_results"]
                if get_similarity_score(r) >= similarity_threshold
            ]

            # Apply deduplication for the table if enabled
            if deduplicate:
                deduplicated_table_results = deduplicate_search_results(
                    filtered_table_results
                )
            else:
                deduplicated_table_results = filtered_table_results
            display_results = deduplicated_table_results[:top_k]

            # Show summary of original vs processed results
            original_count = len(st.session_state["search_results"])
            filtered_count = len(filtered_table_results)
            deduplicated_count = len(deduplicated_table_results)

            processing_text = (
                "After deduplication" if deduplicate else "After filtering"
            )
            st.markdown(f"""
           **Results Summary:**
           - Original results: {original_count}
           - After similarity filtering: {filtered_count}
           - {processing_text}: {deduplicated_count}
           - Displaying top-{min(top_k, deduplicated_count)}: {len(display_results)}
           """)

            results_df = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Base Video ID": slice_id,
                        "Full Slice ID": r.get("row_id", f"Unknown_{i + 1}"),
                        "Link": f"https://lats.bluel3.tools/log-apps/slices/{slice_id}",
                        "Similarity Score": f"{get_similarity_score(r):.4f}",
                        "Timespan": f"{start_ns}, {end_ns}"
                        if start_ns and end_ns
                        else "N/A",
                        "Camera": camera_name if camera_name else "Unknown",
                        "Sensor": r.get("sensor_name", "Unknown"),
                    }
                    for i, r in enumerate(display_results)
                    for slice_id, start_ns, end_ns, camera_name in [
                        parse_row_id(r.get("row_id", f"Unknown_{i + 1}"))
                    ]
                ]
            )

            st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            """
               👈 **Use the sidebar to search videos in two modes: Search by Text or Search by Video**
               - Adjust the Top-K and Similarity Threshold sliders to refine results.
               - Click Search button again after adjustting settings.
               - Initial search would be slow due to data loading.
               - Auth issue may prevent video rendering.


               **🔍 Search by Text:**
               Enter a text query to find videos with similar content.
               - "people walking next to ego vehicle at night"
               - "car turning right at the intersection"
               - "vehicle going through the tunnel"
               - "motorcyclist next to ego vehicle at night"


               **🎬 Search by Video:**
               Select a slice ID from the dropdown to find variations of interesting clips.


               """
        )


def main() -> None:
    """Main application entrypoint."""
    st.set_page_config(
        page_title="alpha Embedding Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()

    st.markdown(
        """
   <div class="main-header">
       <h1>alpha Embedding Search</h1>
   </div>
   """,
        unsafe_allow_html=True,
    )

    query_params = st.query_params
    branch = query_params.get("branch", BRANCH)
    try:
        db_info = load_database_info(branch)
    except Exception as e:
        message = f"❌ Failed to load database: {e}"
        st.error(message)
        st.error("Please check the logs for more details.")
        st.stop()

    with st.sidebar:
        st.markdown(
            '<div class="section-title">🔍 Settings</div>', unsafe_allow_html=True
        )

        current_branch = st.text_input("Database Branch", value=branch)

        viz_option = st.selectbox(
            "Display Mode",
            ["Image Only", "Video Only"],
            index=1,
            help="Choose how to display search results: Image Only shows static frames, Video Only shows video clips",
        )

        deduplicate_results = st.checkbox(
            "Deduplicate Videos",
            value=True,
            help="Remove overlapping segments from the same video, showing only the best representative clip per video",
        )

        top_k = st.slider("Top-K Results", 1, 12, 2)
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.0,
            1.0,
            0.1,
            0.05,
            help="Lower values show more results. Text searches typically need lower thresholds (0.05-0.2) than video searches (0.2-0.5).",
        )

        display_text_search(current_branch, top_k)
        display_video_search(current_branch, db_info.available_slices, top_k)
        display_database_stats(
            db_info.num_slices, db_info.embedding_dim, db_info.sensor_names
        )

    display_retrieval_results(
        top_k, similarity_threshold, viz_option, deduplicate_results
    )


if __name__ == "__main__":  # pragma: no cover
    main()
