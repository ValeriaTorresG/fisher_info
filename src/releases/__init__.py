from typing import Callable, Dict

from argparse import Namespace

from .base import ReleaseConfig


def _factory_edr(args: Namespace) -> ReleaseConfig:
    from . import edr
    return edr.create_config(args)


def _factory_dr1(args: Namespace) -> ReleaseConfig:
    from . import dr1
    return dr1.create_config(args)


def _factory_dr2(args: Namespace) -> ReleaseConfig:
    from . import dr2
    return dr2.create_config(args)


def _factory_yuan23(args: Namespace) -> ReleaseConfig:
    from . import yuan23
    return yuan23.create_config(args)

RELEASE_FACTORIES: Dict[str, Callable[[Namespace], ReleaseConfig]] = {
    'EDR': _factory_edr,
    'DR1': _factory_dr1,
    'DR2': _factory_dr2,
    'YUAN23': _factory_yuan23,
    'SIM': _factory_yuan23,
}

__all__ = ['ReleaseConfig', 'RELEASE_FACTORIES']
