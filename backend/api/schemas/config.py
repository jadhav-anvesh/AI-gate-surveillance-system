from pydantic import BaseModel, Field, field_validator
from typing import List


class FlowConfigRequest(BaseModel):

    line_params: List[List[int]]

    @field_validator("line_params")
    @classmethod
    def validate_lines(cls, value):

        for line in value:

            if len(line) != 4:

                raise ValueError(
                    "Each flow line must contain exactly 4 integers: [x1, y1, x2, y2]"
                )

        return value


class DensityConfigRequest(BaseModel):

    width: int = Field(gt=0)

    height: int = Field(gt=0)


class SpeedConfigRequest(BaseModel):

    source_points: List[List[float]]

    target_points: List[List[float]]

    @field_validator("source_points")
    @classmethod
    def validate_source_points(cls, value):

        if len(value) != 4:

            raise ValueError(
                "Exactly 4 source points are required"
            )

        for point in value:

            if len(point) != 2:

                raise ValueError(
                    "Each source point must be [x, y]"
                )

        return value

    @field_validator("target_points")
    @classmethod
    def validate_target_points(cls, value):

        if len(value) != 4:

            raise ValueError(
                "Exactly 4 target points are required"
            )

        for point in value:

            if len(point) != 2:

                raise ValueError(
                    "Each target point must be [x, y]"
                )

        return value
