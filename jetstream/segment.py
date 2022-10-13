import attr
from metric_config_parser import segment
from mozanalysis import segments


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Segment(segment.Segment):
    """Representation of a segment in Jetstream."""

    def to_mozanalysis_segment(self) -> segments.Segment:
        """Convert the segment to a mozanalysis segment."""
        return segments.Segment(
            name=self.name,
            data_source=segments.SegmentDataSource(
                name=self.data_source.name,
                from_expr=self.data_source.from_expression,
                window_start=self.data_source.window_start,
                window_end=self.data_source.window_end,
                client_id_column=self.data_source.client_id_column,
                submission_date_column=self.data_source.submission_date_column,
                default_dataset=self.data_source.default_dataset,
            ),
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
        )

    @classmethod
    def from_segment_config(cls, segment_config: segment.Segment) -> "Segment":
        """Create a metric class instance from a metric config."""
        args = attr.asdict(segment_config)
        args["data_source"] = segment.SegmentDataSource(**attr.asdict(segment_config.data_source))
        return cls(**args)
