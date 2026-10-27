"""Pure WIA authoring commands. Run: python -m wia.authoring.commands.build_wia_registry --help.

Historical runtime inventories and calibrations use WIA_ARTIFACT_ROOT, which
defaults to the WIA checkout's outputs directory. They are loaded on demand and
are never downloaded. Existing explicit licensed scene and asset paths remain
external inputs. Importing commands does not generate benchmark artifacts.
"""
