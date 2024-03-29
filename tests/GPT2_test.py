import unittest

import torch

from src.LLM.GPT2 import GPT2
from tests.LLM_base import TestLLMBase


class TestGPT2(TestLLMBase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

        cls.model_name = 'gpt2'
        cls.batch_size = 8
        cls.model = GPT2(model_name_or_path=cls.model_name, max_batch_size=cls.batch_size, device=cls.device)

        # https://platform.openai.com/tokenizer with GPT-3 (Legacy)
        cls.test_str_1 = "Hello, world!"
        cls.test_str_token_ids_1 =[15496, 11, 995, 0]

        cls.test_str_2 = "USA means United"
        cls.test_str_2_next_token = 'States'
        cls.test_str_2_next_sentence = 'States of America'
        cls.test_str_token_ids_2 = [14053, 1724, 1578]
        cls.target_id = 1829

    def test_add_positional_encoding(self):
        # Create a mock input tensor
        sequence_length = 11
        mock_input = torch.randn(self.batch_size, sequence_length, self.model.embedding_length, device=self.device)

        # Apply the positional encoding
        output = self.model.add_positional_encoding(mock_input)

        # Check if output is not None
        self.assertIsNotNone(output, "The output of add_positional_encoding is None")

        # Check the shape of the output
        self.assertEqual(
            output.shape,
            mock_input.shape,
            "The output shape of add_positional_encoding is not equal to the input shape."
        )

        # Check that the output and input tensors are on the same device
        self.assertEqual(
            output.device,
            mock_input.device,
            "The output of add_positional_encoding is not on the same device as the input."
        )

        # Check if the positional encoding was added, not replaced
        difference = torch.abs(output - mock_input)
        # The difference should not be zero for all elements
        self.assertFalse(
            torch.all(difference == 0),
            "The output of add_positional_encoding is the same as the input, positional encoding may not have been added."
        )





if __name__ == '__main__':
    unittest.main()