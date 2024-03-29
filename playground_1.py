import torch

from src.LLM.factory import llm_factory
from src.Datasets.factory import trex_star_factory, trex_bite_factory
#
# dataset_trexstar_lite = trex_star_factory("TRExStarLite")
# print(f"TRExStarLite:all", len(dataset_trexstar_lite))
#
dataset_trexstar = trex_star_factory("TRExStar")
print(f"TRExStar:all", len(dataset_trexstar))
#
# bite_lite_train, bite_lite_validate, bite_lite_test = trex_bite_factory('TRExBiteLite')
# print(f"TRExBiteLite:train", len(bite_lite_train))
# print(f"TRExBiteLite:validation", len(bite_lite_test))
# print(f"TRExBiteLite:test", len(bite_lite_validate))
#
# bite_train, bite_validate, bite_test = trex_bite_factory('TRExBite')
# print(f"TRExBite:train", len(bite_train))
# print(f"TRExBite:validation", len(bite_test))
# print(f"TRExBite:test", len(bite_validate))





llm = llm_factory(
    "gpt-2",
    "gpt2",
    batch_size=1,
    device=torch.device("cpu"),
    bits=2
)
# wgb = WikidataGraphBuilder(llm)

