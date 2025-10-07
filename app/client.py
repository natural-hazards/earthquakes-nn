import obspy

from obspy.clients.fdsn import RoutingClient
from obspy.core.utcdatetime import UTCDateTime

class EidaRoutingClient:

    @staticmethod
    def get_waveform(
        station_code: str,
        time_start: UTCDateTime,
        time_end: UTCDateTime,
        channel: str,
    ) -> obspy.Stream:
        client = RoutingClient("eida-routing")
        stream = client.get_waveforms(
            network='CZ',
            station=station_code,
            channel=f'{channel}*',
            starttime=time_start,
            endtime=time_end
        )

        return stream.slice(time_start, time_end)

    @staticmethod
    def get_stations(
        time_start: UTCDateTime,
        time_end: UTCDateTime,
    ) -> list:
        client = RoutingClient("eida-routing")
        inventory = client.get_stations(
            network='CZ',
            starttime=time_start,
            endtime=time_end,
            level='station'
        )
        return inventory.get_contents()['stations']

    @staticmethod
    def get_stations_channels(
        station_code: str,
        time_start: UTCDateTime,
        time_end: UTCDateTime,
    ) -> list:
        client = RoutingClient("eida-routing")
        inventory = client.get_stations(
            network='CZ',
            station=station_code,
            starttime=time_start,
            endtime=time_end,
            level='channel'
        )

        channels: list = inventory.get_contents()['channels']
        channels = sorted(
            set([ch[-3:-1] for ch in channels])
        )

        return channels
