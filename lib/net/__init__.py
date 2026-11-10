from net.network import Network
from net.bbn_network import BBNNetwork


def build_network(cfg, mode="train", num_classes=1000):
    network_type = str(getattr(cfg.NETWORK, "TYPE", "standard")).lower()
    if network_type == "bbn":
        return BBNNetwork(cfg, mode=mode, num_classes=num_classes)
    if network_type == "standard":
        return Network(cfg, mode=mode, num_classes=num_classes)
    raise ValueError(f"Unsupported NETWORK.TYPE: {cfg.NETWORK.TYPE}")

__all__ = [
    'Network', 'BBNNetwork', 'build_network',
]
