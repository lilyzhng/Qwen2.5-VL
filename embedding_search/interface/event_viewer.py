import logging
from collections.abc import Sequence
from dataclasses import Field, dataclass, fields
from typing import Annotated, Any, Final, Literal, Optional, get_args, get_origin


import streamlit as st
from streamlit.delta_generator import DeltaGenerator


_LOGGER: Final = logging.getLogger(__name__)




@dataclass
class LatitubeConfig:
   """Configuration for the event viewer grid."""


   cols: Annotated[int, {"min": 1, "max": 6, "label": "Columns"}] = 3
   height: Annotated[int, {"min": 200, "max": 800, "label": "Height (px)"}] = 500
   autoplay: Annotated[bool, {"label": "Autoplay"}] = True
   loop: Annotated[bool, {"label": "Loop"}] = True
   kind: Annotated[Literal["preview", "potato"], {"label": "View Type"}] = "preview"




def _get_field_metadata(field: Field[Any]) -> Any:
   """Extract metadata from field annotation so that adding fields to LatitubeConfig doesn't need other changes.


   Args:
       field: The dataclass field to extract metadata from


   Returns:
       Dictionary containing the field's metadata
   """
   if hasattr(field.type, "__metadata__") and field.type.__metadata__:
       return field.type.__metadata__[0]
   return {}




def _create_widget_for_field(field: Field[Any], value: Any, column: DeltaGenerator) -> Any:
   """Create appropriate Streamlat widget for a dataclass field based on type and metadata.


   Args:
       field: The dataclass field to create a widget for
       value: The current value of the field
       column: The Streamlat column to place the widget in


   Returns:
       The new value from the widget
   """
   metadata = _get_field_metadata(field)
   label = metadata.get("label", field.name.title())


   field_type = field.type
   if get_origin(field_type) is Annotated:
       field_type = get_args(field_type)[0]


   with column:
       if field_type == int:
           min_val: int = metadata.get("min", 0)
           max_val: int = metadata.get("max", 100)
           return st.slider(label, min_value=min_val, max_value=max_val, value=value)


       elif field_type == bool:
           st.markdown(f"{label}")  # Label above checkbox for consistency
           return st.checkbox(label, value=value, key=f"checkbox_{field.name}", label_visibility="collapsed")


       elif get_origin(field_type) is Literal:
           options: list[str] = list(get_args(field_type))
           index: int = options.index(value) if value in options else 0
           return st.selectbox(label, options=options, index=index)


       else:
           # Fallback for other types
           return st.text_input(label, value=str(value))




def _show_filters(config: Optional[LatitubeConfig] = None) -> LatitubeConfig:
   """Show grid configuration filters dynamically based on dataclass fields.


   Args:
       config: Default configuration to use


   Returns:
       LatitubeConfig with user-selected values
   """
   config = config or LatitubeConfig()
   dataclass_fields: tuple[Field[Any], ...] = fields(config)
   columns: list[DeltaGenerator] = st.columns(
       len(dataclass_fields),
       gap="medium",
       border=True,
   )
   field_values: dict[str, Any] = {}


   for i, field in enumerate(dataclass_fields):
       current_value: Any = getattr(config, field.name)
       new_value: Any = _create_widget_for_field(field, current_value, columns[i])
       field_values[field.name] = new_value


   return LatitubeConfig(**field_values)




def _show_iframe(
   event_id: str,
   config: LatitubeConfig,
   show_titles: bool = True,
) -> None:
   """Show a single LaTS event iframe.


   Args:
       event_id: The LaTS event ID to display
       config: Configuration for the iframe
       show_titles: Whether to show the event ID title above the iframe
   """
   url: str = "https://lats.bluel3.tools/latitube/"
   params: list[str] = [f"id={event_id}"]
   params.append(f"autoplay={str(config.autoplay).lower()}")
   params.append(f"loop={str(config.loop).lower()}")
   params.append(f"kind={config.kind}")


   if params:
       url += "?" + "&".join(params)


   if show_titles:
       st.markdown(f"**[`{event_id}`](https://to/lats/{event_id})**")


   _LOGGER.warning("Loading LaTS event with URL: %s", url)
   iframe_html: str = f"""
   <iframe
       src="{url}"
       width="100%"
       height="{config.height}px"
       frameborder="0"
       sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
       referrerpolicy="strict-origin-when-cross-origin"
       style="border: 1px solid #ddd; border-radius: 4px;">
   </iframe>
   """
   st.components.v1.html(iframe_html, height=config.height + 10)




def latitube(
   event_ids: Sequence[str],
   config: Optional[LatitubeConfig] = None,
   show_titles: bool = True,
   allow_user_configurable_grid: bool = False,
) -> None:
   """Display a grid of LaTS event iFrames in Streamlit.


   Args:
       event_ids: List of LaTS event IDs to display
       config: Configuration for the event viewer (uses defaults if None)
       show_titles: Whether to show event ID titles above each iframe (default: True)
       allow_user_configurable_grid: Whether to show grid configuration controls (default: False)
   """
   if not event_ids:
       st.warning("No event IDs provided")
       return


   config = config or LatitubeConfig()


   if allow_user_configurable_grid:
       config = _show_filters(config)


   if config.cols < 1:
       config = LatitubeConfig(
           cols=1, height=config.height, autoplay=config.autoplay, loop=config.loop, kind=config.kind
       )


   rows: int = (len(event_ids) + config.cols - 1) // config.cols


   st.markdown(f"📺 **Displaying {len(event_ids)} LaTS events in {rows}x{config.cols} grid**")


   for row in range(rows):
       columns: list[DeltaGenerator] = st.columns(config.cols)


       for col in range(config.cols):
           idx: int = row * config.cols + col


           if idx < len(event_ids):
               event_id: str = event_ids[idx]


               with columns[col]:
                   _show_iframe(event_id, config, show_titles)