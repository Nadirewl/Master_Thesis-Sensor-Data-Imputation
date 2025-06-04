import functools
import logging as logger

import pandas as pd
from babayaga import BabaYagaClient
from babayaga.errors import (
    ConfigurationError,
    DocumentError,
    FileError,
    MetadataError,
    ReferenceCurveError,
    SITranslationError,
)

# In future the mapping int->tag should be in babayaga client
REFERENCE_MODE_MAP = {
    1: ["HFO_TII"],
    2: ["HFO_TII_TCCO", "HFO_TII"],  # First one preferred over second one !
    3: ["HFO_TIII"],
    4: ["HFO_TIII"],
    5: ["DF_TII"],
    6: ["DF_TII_TCCO", "DF_TII"],  # First one preferred over second one !
    7: ["DF_TIII"],
    8: ["DF_TIII"],
    9: ["HFO_TII_ECOEGR"],
    10: [],
    11: [],
    12: [],
    13: ["DF_TII_ECOEGR"],
    14: [],
    15: [],
    16: [],
}

# Expected opv names in entry and mapping to names expected in uc90core
CORE_EXPECTED_OPV_MAP = [
    ("te_exh_turb_in_iso", "reference_curve_turbine_in"),
    ("fr_tc_iso", "reference_curve_tc_speed"),
    ("pr_air_scav_iso", "reference_curve_pscav_control"),
    ("pr_cyl_comp_mv_iso", "reference_curve_pcomp_control"),
    ("pr_cyl_max_mv_iso", "reference_curve_pmax_control"),
]


def _cache(func):
    """Cache function results for an objects lifetime, without cyclic references (pylint:W1518)"""

    UNCACHED = object()  # This is a unique object to represent an uncached value

    @functools.wraps(func)
    def wrapper(self):
        cached_name = f"_{func.__name__}_cache"
        if getattr(self, cached_name, UNCACHED) is UNCACHED:
            setattr(self, cached_name, func(self))
        return getattr(self, cached_name)

    return wrapper


class BabayagaWrapper:
    """Retrieves and processes data retrieved from Babayaga Client"""

    def __init__(
        self,
        engine_uuid: str,
        baba_client: BabaYagaClient | None = None,
        reference_mode: int | None = None,
    ):
        assert isinstance(reference_mode, (int, type(None)))
        self._reference_mode: int | None = reference_mode
        self._engine_uuid = engine_uuid
        self._baba_client = baba_client or self._get_baba_yaga_client()

    @property
    @_cache
    def fr_eng_smcr(self) -> float | None:
        """Get maximum engine speed @ service rating

        Returns:
            Maximum engine speed @ service rating
        """
        if self._baba_client is None:
            return None
        try:
            return self._baba_client.get_metadata().fr_eng_smcr
        except (MetadataError,):
            logger.critical(f"Retrieval of fr_eng_smcr failed for {self._engine_uuid}")
            return None

    @property
    @_cache
    def reference_curves(self) -> dict:
        """Extract curves from Babayaga, return as a dictionary for uc90core

        Returns:
            Dict with {opv: curves_as_Series}
        """
        curve_series_dict = {}
        if self._baba_client is None:
            return curve_series_dict
        try:
            curve_tag_to_use = self._tag_to_use(self._baba_client.reference_modes)
            if curve_tag_to_use is None:
                raise KeyError

            # Load values for measurement numbers
            load_values = self._baba_client.get_si_load_index()[curve_tag_to_use]
            raw_curves_df = self._baba_client.si_curves[curve_tag_to_use]["raw"]
            for opv in CORE_EXPECTED_OPV_MAP:
                if opv[0] in raw_curves_df.columns:
                    curve_series_dict[opv[0]] = pd.Series(
                        index=load_values.values, data=raw_curves_df[opv[0]].to_list()
                    )

            # Replace the keys as the reference curves are expected to have in uc90core
            self.replace_keys_for_core(curve_series_dict)

        except (
            ConfigurationError,
            MetadataError,
            DocumentError,
            ReferenceCurveError,
            FileError,
            SITranslationError,
            KeyError,
            AttributeError,
        ) as e:
            logger.critical(
                f"Retrieval of ref curves failed for {self._engine_uuid} and reference_mode {self._reference_mode}, error: {str(e)}"
            )

        return curve_series_dict

    @property
    @_cache
    def min_load_curves(self) -> float | None:
        if not self.reference_curves:
            return None
        return min(s.index.min() for k, s in self.reference_curves.items())

    def _get_baba_yaga_client(self):
        try:
            return BabaYagaClient(self._engine_uuid)
        except (ConfigurationError, DocumentError):
            logger.critical(f"Retrieval of BabaYagaClient failed for {self._engine_uuid}")
        return None

    def _tag_to_use(self, available_tags: list) -> str | None:
        """Get tag to use for the current reference mode"""
        for tag in REFERENCE_MODE_MAP[self._reference_mode]:
            if tag in available_tags:
                return tag
        return None

    @staticmethod
    def replace_keys_for_core(curve_series_dict: dict) -> dict:
        """Replace input dict keys for core values expected in UC90 Core"""
        for old_key, new_key in CORE_EXPECTED_OPV_MAP:
            if old_key in curve_series_dict:
                curve_series_dict[new_key] = curve_series_dict.pop(old_key)
        return curve_series_dict
