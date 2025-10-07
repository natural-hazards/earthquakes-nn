import streamlit as st
import altair as alt

import numpy as np
import pandas as pd

from obspy.core import UTCDateTime
from client import EidaRoutingClient

SERVER_DELAY = 310

class Page:

    def __init__(
        self
    ) -> None:
        self.client = EidaRoutingClient()

    @st.fragment(run_every=1.5)
    def _waveform_fragment(
        self,
        station_code: str,
        channel: str,
    ) -> None:
        stream = self.client.get_waveform(
            station_code=station_code,
            channel=channel,
            time_start=UTCDateTime.now() - SERVER_DELAY,
            time_end=UTCDateTime.now()
        )

        for tr in stream:
            with st.container(border=True): self.render_trace_info(tr)

    @staticmethod
    def render_trace_info(
        tr
    ) -> None:
        with st.expander(f"Trace info: {tr.id}"):
            st.write(tr.stats)

        times = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        timestamps = [UTCDateTime.now() - SERVER_DELAY + t for t in times]

        df = pd.DataFrame({"Time": timestamps, "Amplitude": tr.data})
        chart = alt.Chart(df).mark_line().encode(
            x="Time:T",
            y=alt.Y("Amplitude:Q", scale=alt.Scale(domain=[-4000, 4000]))  # Set min and max
        ).properties(
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

    def render(
        self,
    ) -> None:
        station: str = st.selectbox(
            "Station",
            options=self.client.get_stations(
                time_start=UTCDateTime.now() - 86400,
                time_end=UTCDateTime()
            )
        )
        station_code = station.split(' ')[0].split('.')[1]

        channel: str = st.selectbox(
            'Channel',
            options=self.client.get_stations_channels(
                station_code=station_code,
                time_start=UTCDateTime.now() - 86400,
                time_end=UTCDateTime()
            )
        )
        channel = channel.split('.')[-1]

        self._waveform_fragment(
            station_code=station_code,
            channel=channel,
        )

Page().render()