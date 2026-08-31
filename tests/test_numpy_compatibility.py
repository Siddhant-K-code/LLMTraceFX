from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pyarrow as pa
from packaging.version import Version


def test_numpy_2_dataframe_arrow_and_plotly_interoperability() -> None:
    assert Version(np.__version__) >= Version("2.2.6")
    assert Version(pd.__version__) >= Version("2.2.2")
    assert Version(plotly.__version__) >= Version("7.0.0")

    latencies = np.asarray([1.25, 2.5, 5.0], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "token_id": np.arange(latencies.size, dtype=np.int64),
            "latency_ms": latencies,
            "timestamp": [
                datetime(2026, 8, 31, minute, tzinfo=timezone.utc)
                for minute in range(latencies.size)
            ],
        }
    )

    table = pa.Table.from_pandas(frame, preserve_index=False)
    restored = table["latency_ms"].to_numpy()
    figure = px.line(frame, x="token_id", y="latency_ms")
    csv_frame = pd.read_csv(StringIO(frame.to_csv(index=False)))

    assert frame["timestamp"].dt.tz is timezone.utc
    assert csv_frame.shape == frame.shape
    assert restored.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(restored, latencies)
    np.testing.assert_array_equal(figure.data[0].y, latencies)
