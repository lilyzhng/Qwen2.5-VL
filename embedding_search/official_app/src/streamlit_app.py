# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from typing import Union, Optional
import pandas as pd
import plotly.express as px
import faiss
import numpy as np
from transformers import AutoModel, AutoProcessor
import torch
from datetime import datetime


class SelectedIndex:
    def __init__(self, idx) -> None:
        self.idx = int(idx)
        self.timestamp = datetime.now()

    def __eq__(self, value: Union["SelectedIndex", int]) -> bool:
        if isinstance(value, SelectedIndex):
            return self.idx == value.idx
        return self.idx == int(value)

    def __ne__(self, value: Union["SelectedIndex", int]) -> bool:
        return not self.__eq__(value)

    def is_valid(self) -> bool:
        return self.idx >= 0


@st.cache_data
def load_data(path: str):
    df = pd.read_parquet(path)
    embs = np.stack(df["embedding"].tolist()).astype("float32")
    faiss.normalize_L2(embs)
    D = embs.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(embs)
    return df, index, embs


def load_model() -> tuple[AutoModel, AutoProcessor]:
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = AutoProcessor.from_pretrained(
            "nvidia/Cosmos-Embed1-224p", trust_remote_code=True, token=True,
        )
    if "model" not in st.session_state:
        model = AutoModel.from_pretrained(
            "nvidia/Cosmos-Embed1-224p", trust_remote_code=True, token=True,
        )
        model.eval()
        st.session_state.model = model
    return st.session_state.model, st.session_state.preprocessor


def preview_video(df, idx, slot, height=420, margin_top=30, autoplay=True, title=None) -> None:
    if title:
        slot.markdown(f"### {title}")
    start = int(df.loc[idx, "span_start"])
    end = int(df.loc[idx, "span_end"])
    youtube_id = df.loc[idx, "youtube_id"]
    url = f"https://www.youtube.com/embed/{youtube_id}?start={start}&end={end}"
    sep = "?" if "?" not in url else "&"
    params = f"{sep}mute=1&rel=0"
    if autoplay:
        params += "&autoplay=1"
    slot.markdown(
        f'''
        <div style="margin-top:{margin_top}px">
          <iframe width="100%" height="{height}"
                  src="{url}{params}"
                  frameborder="0"
                  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                  allow="autoplay; fullscreen" allowfullscreen>
          </iframe>
        </div>
        ''',
        unsafe_allow_html=True
    )


def get_nearest_ids(vec, k=5, ignore_self=True) -> list:
    q = vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    topk = k + 1 if ignore_self else k
    _, I = index.search(q, topk)
    ids = I[0]
    return ids[1:].tolist() if ignore_self else ids.tolist()


def get_most_recent_selection() -> tuple[Optional[int], str]:
    if st.session_state.text_selection.is_valid() and st.session_state.click_selection.is_valid():
        if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
            return st.session_state.text_selection.idx, "text"
        return st.session_state.click_selection.idx, "click"
    if st.session_state.text_selection.is_valid():
        return st.session_state.text_selection.idx, "text"
    if st.session_state.click_selection.is_valid():
        return st.session_state.click_selection.idx, "click"
    return None, ""


def reset_state() -> None:
    if "text_selection" not in st.session_state:
        st.session_state.text_selection = SelectedIndex(-1)
    if "click_selection" not in st.session_state:
        st.session_state.click_selection = SelectedIndex(-1)
    if "text_query" not in st.session_state:
        st.session_state.text_query = ""

# ─── App setup ────────────────────────────────────────────────────

st.set_page_config(layout="wide")
reset_state()
model, preprocessor = load_model()
file_map = {"kinetics700 (val)": "src/kinetics700_val.parquet", "opendv (val)": "src/opendrive_val.parquet"}
st.title("🔍 Search with Cosmos-Embed1")

col1, col2 = st.columns([2,2])
with col1:
    dataset = st.selectbox("Select dataset", list(file_map.keys()), on_change=reset_state)
df, index, embs = load_data(file_map[dataset])

# initialize session state
if "text_selection" not in st.session_state:
    st.session_state.text_selection = SelectedIndex(-1)
if "click_selection" not in st.session_state:
    st.session_state.click_selection = SelectedIndex(-1)
if "text_query" not in st.session_state:
    st.session_state.text_query = ""

# ─── Layout ────────────────────────────────────────────────────────

# LEFT: scatter
with col1:
    fig = px.scatter(
        df, x="x", y="y",
        hover_name="tar_key", hover_data=["cluster_id"],
        color="cluster_id", color_continuous_scale="Turbo",
        title="t-SNE projection (click to select)"
    )
    fig.update_layout(
        dragmode="zoom",
        margin=dict(l=5, r=5, t=40, b=5),
        xaxis_title=None, yaxis_title=None,
        coloraxis_colorbar=dict(title="")
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=True, linecolor="black", mirror=True)
    fig.update_layout(annotations=[dict(
        text="k-means cluster", xref="paper", yref="paper",
        x=1.02, y=0.5, textangle=90, showarrow=False
    )])

    most_recent_idx, most_recent_method = get_most_recent_selection()
    if most_recent_idx is not None and most_recent_method == "text":
        x0, y0 = df.iloc[most_recent_idx][["x", "y"]]
        span = 6.0
        fig.update_layout(
            xaxis_range=[x0 - span, x0 + span],
            yaxis_range=[y0 - span, y0 + span],
            transition={"duration": 1},
        )

    click_event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode="points"
    )

# RIGHT: text input & preview
with col2:
    if click_event and click_event.get("selection", {}).get("point_indices"):
        curr_click = click_event["selection"]["point_indices"][0]
        if curr_click != st.session_state.click_selection:
            # new click so update the previous selection and wipe any text query
            st.session_state.click_selection = SelectedIndex(curr_click)
            st.session_state.text_query = ""

    # text input (will pick up cleared or existing text)
    text_query = st.text_input(
        "Search via text",
        key="text_query",
        help="Type a query and press Enter"
    )

    # if user typed text (and pressed Enter), override selection
    if text_query:
        with torch.no_grad():
            model_input = preprocessor(
                text=[text_query],
                return_tensors="pt"
            )
            emb_out = model.get_text_embeddings(
                input_ids=model_input["input_ids"],
                attention_mask=model_input["attention_mask"]
            ).text_proj.cpu().numpy()
        idx_text, = get_nearest_ids(emb_out, k=1, ignore_self=False)
        if st.session_state.text_selection != idx_text:
            # new text so update the previous selection and wipe any text query
            st.session_state.text_selection = SelectedIndex(idx_text)
            st.rerun()

    # main preview
    preview_slot = st.empty()
    most_recent, most_recent_modality = get_most_recent_selection()
    if most_recent is not None:
        preview_video(df, most_recent, preview_slot)
    else:
        preview_slot.write("⏳ Waiting for selection…")

# BOTTOM: 5 nearest neighbors
st.markdown("### 🎬 5 Closest Videos")
if most_recent is not None:
    ignore_self = most_recent_modality == "click"
    nn_ids = get_nearest_ids(embs[most_recent], k=5, ignore_self=ignore_self)
    cols = st.columns(5)
    for c, nid in zip(cols, nn_ids):
        preview_video(df, nid, c, height=180, margin_top=5, autoplay=False)
else:
    st.write("Use a click or a text query above to list neighbors.")
