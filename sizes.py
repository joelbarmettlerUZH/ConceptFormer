from Datasets.factory import trex_star_factory, trex_bite_factory, trirex_factory, web_qsp_factory

dataset_trexstar_lite = trex_star_factory("TRExStarLite")
print(f"TRExStarLite:all", len(dataset_trexstar_lite))

dataset_trexstar = trex_star_factory("TRExStar")
print(f"TRExStar:all", len(dataset_trexstar))

bite_lite_train, bite_lite_validate, bite_lite_test = trex_bite_factory('TRExBiteLite')
print(f"TRExBiteLite:train", len(bite_lite_train))
print(f"TRExBiteLite:validation", len(bite_lite_validate))
print(f"TRExBiteLite:test", len(bite_lite_test))

trirex_lite_train, trirex_lite_validate, trirex_lite_test = trirex_factory('TriRExLite')
print(f"TriRExLite:train", len(trirex_lite_train))
print(f"TriRExLite:validation", len(trirex_lite_validate))
print(f"TriRExLite:test", len(trirex_lite_test))

bite_train, bite_validate, bite_test = trex_bite_factory('TRExBite')
print(f"TRExBite:train", len(bite_train))
print(f"TRExBite:validation", len(bite_validate))
print(f"TRExBite:test", len(bite_test))

trirex_train, trirex_validate, trirex_test = trirex_factory('TriREx')
print(f"TriREx:train", len(trirex_train))
print(f"TriREx:validation", len(trirex_validate))
print(f"TriREx:test", len(trirex_test))

sentence_test_dataset, graphs = web_qsp_factory()
print(f"WebQSPSentences:test", len(sentence_test_dataset))
print(f"WebQSPStar:all", len(graphs))
