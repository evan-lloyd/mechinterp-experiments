import os
import tempfile
import uuid
import zipfile
from collections import defaultdict
from io import StringIO
from typing import TYPE_CHECKING, Sequence, Tuple

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, SVG, display
from safetensors import safe_open
from safetensors.numpy import save_file as save_file_numpy
from safetensors.torch import save_file as save_file_torch
from transformers import AutoModel, AutoTokenizer, TextStreamer

if TYPE_CHECKING:
    from .sae import SAE
    from .training import SAECheckpoint, TrainingResult

# TinyStories33M max_position_embeddings
MAX_GENERATION = 2048


def save_checkpoint(checkpoint: "SAECheckpoint", out_file: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save SAE weights using safetensors
        if checkpoint.sae is not None:
            sae_path = os.path.join(tmpdir, "sae_weights.safetensors")
            save_file_torch(checkpoint.sae.state_dict(), sae_path)

            # Save SAE config using cloudpickle
            config_path = os.path.join(tmpdir, "sae_config.cloudpickle")
            with open(config_path, "wb") as f:
                cloudpickle.dump(checkpoint.sae.config, f)

        # Save metrics (np.ndarray elements) using safetensors
        metrics_dict = {
            "step_tokens_trained": checkpoint.step_tokens_trained,
        }
        for key, value in checkpoint.step_metrics.items():
            metrics_dict[f"step_metrics.{key}"] = value

        metrics_path = os.path.join(tmpdir, "metrics.safetensors")
        save_file_numpy(metrics_dict, metrics_path)

        # Save scalar values using cloudpickle
        scalars_path = os.path.join(tmpdir, "scalars.cloudpickle")
        with open(scalars_path, "wb") as f:
            cloudpickle.dump(
                {"total_tokens_trained": checkpoint.total_tokens_trained}, f
            )

        # Create zip file
        with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in os.listdir(tmpdir):
                filepath = os.path.join(tmpdir, filename)
                zf.write(filepath, filename)


def load_checkpoint(in_file: str) -> "SAECheckpoint":
    from .sae import SAE
    from .training import SAECheckpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract zip file
        with zipfile.ZipFile(in_file, "r") as zf:
            zf.extractall(tmpdir)

        # Load scalar values
        scalars_path = os.path.join(tmpdir, "scalars.cloudpickle")
        with open(scalars_path, "rb") as f:
            scalars = cloudpickle.load(f)

        # Load metrics
        metrics_path = os.path.join(tmpdir, "metrics.safetensors")
        with safe_open(metrics_path, framework="numpy") as f:
            step_tokens_trained = f.get_tensor("step_tokens_trained")
            step_metrics = {}
            for key in f.keys():
                if key.startswith("step_metrics."):
                    metric_name = key[len("step_metrics.") :]
                    step_metrics[metric_name] = f.get_tensor(key)

        # Load SAE if it exists
        sae = None
        sae_path = os.path.join(tmpdir, "sae_weights.safetensors")
        config_path = os.path.join(tmpdir, "sae_config.cloudpickle")
        if os.path.exists(sae_path) and os.path.exists(config_path):
            with open(config_path, "rb") as f:
                sae_config = cloudpickle.load(f)
            sae = SAE(sae_config)
            with safe_open(sae_path, framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            sae.load_state_dict(state_dict, assign=True)

        checkpoint = SAECheckpoint(
            total_tokens_trained=scalars["total_tokens_trained"],
            step_tokens_trained=step_tokens_trained,
            step_metrics=step_metrics,
            sae=sae,
        )

        return checkpoint


def save_training_result(result: "TrainingResult", out_dir: str):
    """Save a TrainingResult to the given directory.

    Each checkpoint is saved to its own file, named based on the layer
    and number of training tokens for that checkpoint.
    """
    os.makedirs(out_dir, exist_ok=True)

    for layer, checkpoints in result.items():
        for checkpoint in checkpoints:
            filename = (
                f"layer_{layer}_tokens_{checkpoint.total_tokens_trained}.checkpoint"
            )
            filepath = os.path.join(out_dir, filename)
            save_checkpoint(checkpoint, filepath)


def load_training_result(from_dir: str) -> "TrainingResult":
    """Load a TrainingResult from the given directory.

    Loads all checkpoint files and reconstructs the TrainingResult structure.
    """
    from .training import TrainingResult

    # Find all checkpoint files and group by layer
    checkpoints_by_layer = defaultdict(list)
    for filename in os.listdir(from_dir):
        if filename.endswith(".checkpoint"):
            # Parse filename: layer_{layer}_tokens_{tokens}.checkpoint
            parts = filename.replace(".checkpoint", "").split("_")
            layer = int(parts[1])
            tokens = int(parts[3])
            filepath = os.path.join(from_dir, filename)
            checkpoint = load_checkpoint(filepath)
            checkpoints_by_layer[layer].append((tokens, checkpoint))

    # Sort checkpoints by tokens trained and extract SAEs for initialization
    saes = {}
    for layer, checkpoint_list in checkpoints_by_layer.items():
        checkpoint_list.sort(key=lambda x: x[0])
        # Use the first checkpoint's SAE for initialization
        if checkpoint_list:
            saes[layer] = checkpoint_list[0][1].sae

    # Create TrainingResult and populate with checkpoints
    result = TrainingResult(saes)
    for layer, checkpoint_list in checkpoints_by_layer.items():
        # Replace the initial checkpoint list with loaded checkpoints
        result._layer_results[layer] = [cp for _, cp in checkpoint_list]

    return result


def clone_sae(sae: "SAE", to_device: str | None = None):
    from .sae import SAE

    result = SAE(sae.config)
    result.init_weights(sae, to_device)

    return result


def generate(
    inputs,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    stream=True,
    stream_callback=None,
    **kwargs,
):
    if isinstance(inputs, str):
        inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    class TextStreamerWithCallback(TextStreamer):
        def put(self, value):
            super().put(value)

            if stream_callback:
                stream_callback(value[0])

    streamer = TextStreamerWithCallback(tokenizer, skip_prompt=True) if stream else None

    try:
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_length=MAX_GENERATION,
                streamer=streamer,
                tokenizer=tokenizer,
                **kwargs,
            )
    except KeyboardInterrupt:
        print("\n*** Generation aborted by user ***")


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def current_plot_to_svg(filename: str | None = None, plot_dir: str = "/tmp"):
    plot_svg = StringIO()
    plt.savefig(plot_svg, format="svg")
    plt.close()
    plot_svg.seek(0)
    d = display(SVG(plot_svg.read()))

    if filename:
        plot_svg.seek(0)
        ensure_directory(plot_dir)
        open(f"{plot_dir}/{filename}.svg", "w").write(plot_svg.read())

    return d


def ensure_tensor(
    maybe_tensor: Tuple[torch.Tensor, ...] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(maybe_tensor, tuple):
        return maybe_tensor[0]
    return maybe_tensor


def _sortable_table_html(
    table_id: str, sort_fn: str, header_row: str, data_rows: list[str]
) -> str:
    """Build the full HTML + inline script for a sortable table."""
    return f"""
<table id="{table_id}" border="1" style="border-collapse: collapse;">
{header_row}
{"".join(data_rows)}
</table>
<script>
(function() {{
    var table = document.getElementById("{table_id}");
    var sortDirections = {{}};

    window.{sort_fn} = function(columnIndex) {{
        var rows = Array.from(table.rows).slice(1);
        var dir = sortDirections[columnIndex] || 'asc';

        rows.sort(function(a, b) {{
            var aVal = a.cells[columnIndex].innerText;
            var bVal = b.cells[columnIndex].innerText;
            var aNum = parseFloat(aVal);
            var bNum = parseFloat(bVal);

            if (!isNaN(aNum) && !isNaN(bNum)) {{
                return dir === 'asc' ? aNum - bNum : bNum - aNum;
            }}
            return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        }});

        rows.forEach(function(row) {{ table.appendChild(row); }});
        sortDirections[columnIndex] = dir === 'asc' ? 'desc' : 'asc';
    }};
}})();
</script>
"""


def render_sortable_html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    table_id: str | None = None,
) -> str:
    """Render a sortable HTML table from headers and row data.

    Args:
        headers: Column header labels.
        rows: List of row tuples/lists. Each row should have one value per column.
              Values are rendered as-is; numbers and strings both work for sorting.
        table_id: Optional unique ID for the table element. If not provided,
                  a UUID-based ID is generated.

    Returns:
        Complete HTML string including the table and inline sort script.
        Use with IPython: ``display(HTML(render_sortable_html_table(...)))``.
    """
    if table_id is None:
        table_id = f"sortable_table_{uuid.uuid4().hex[:8]}"
    sort_fn = f"sortTable_{table_id.replace('-', '_')}"

    header_cells = "".join(
        f'<th onclick="{sort_fn}({i})" style="cursor:pointer">{h} ⇅</th>'
        for i, h in enumerate(headers)
    )
    header_row = f"<tr>{header_cells}</tr>"

    data_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        data_rows.append(f"<tr>{cells}</tr>")

    return _sortable_table_html(table_id, sort_fn, header_row, data_rows)


def display_sortable_html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    table_id: str | None = None,
):
    """Display a sortable HTML table in a Jupyter notebook.

    Convenience wrapper around ``render_sortable_html_table`` that displays
    the result. Same arguments as ``render_sortable_html_table``.
    """
    display(HTML(render_sortable_html_table(headers, rows, table_id)))
