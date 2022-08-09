from jetstream_config_parser import segment
from mozanalysis.segments import Segment, SegmentDataSource


class Segment(segment.Segment):
    """Representation of a segment in Jetstream."""

    def to_mozanalysis_segment(self) -> Segment:
        """Convert the segment to a mozanalysis segment."""
        return Segment(
            name=self.name,
            data_source=SegmentDataSource(
                name=self.data_source.name,
                from_expr=self.data_source.from_expression,
                window_start=self.data_source.window_start,
                window_end=self.data_source.window_end,
                client_id_column=self.data_source.client_id,
                submission_date_column=self.data_source.submission_date_column,
                default_dataset=self.data_source.default_dataset,
            ),
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
        )
