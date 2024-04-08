import unittest
import time

import torch



class TestLLMBase(unittest.TestCase):

    def test_add_positional_encoding(self):
        raise NotImplementedError("Subclasses should implement this!")

    def test_initialization(self):
        # Check if model and tokenizer are initialized correctly
        self.assertIsNotNone(self.model._model, "Model should be initialized.")
        self.assertIsNotNone(self.model._tokenizer, "Tokenizer should be initialized.")

        # Check if the model is on the correct device
        self.assertEqual(next(self.model._model.parameters()).device, self.device,
                         "Model parameters should be on the specified device.")

        # Verify the embedding length
        expected_embedding_length = self.model.embedding_length
        self.assertEqual(self.model.embedding_length, expected_embedding_length,
                         "Embedding length should match model configuration.")

        # Verify batch size initialization
        self.assertEqual(self.model.max_batch_size, self.batch_size,
                         "Max batch size should be set correctly.")

        # Check if model's token embeddings are resized
        # self.assertEqual(self.model._model.Config.hidden_size, len(self.model._tokenizer),
        #                  "The model's token embeddings should be resized to match the tokenizer's length.")



    def test_text_to_input_ids(self):
        # Convert text to input IDs using the method
        input_ids = self.model.text_to_input_ids(self.test_str_1)

        # Check the device is correct
        self.assertEqual(input_ids.device, self.device, "The input IDs are not on the correct device.")

        # Check the shape of the tensor (should be (1, sequence_length))
        self.assertEqual(input_ids.shape[0], 1, "The input IDs tensor should have one row.")
        self.assertTrue(input_ids.shape[1] <= self.model.max_sequence_length, f"The input IDs tensor should not exceed the max_length of {self.model.max_sequence_length}.")

        # Check the content of the input IDs tensor
        actual_input_ids = input_ids[0].tolist()  # Flatten the tensor to list for easy comparison
        self.assertEqual(actual_input_ids[:len(self.test_str_token_ids_1)], self.test_str_token_ids_1,
                         "The input IDs do not match the expected token IDs.")

        # Check if truncation works (assuming 'Hello, world!' is shorter than the maximum length)
        long_string = self.test_str_1 * 1000  # create a long string to force truncation
        truncated_input_ids = self.model.text_to_input_ids(long_string)
        self.assertTrue(truncated_input_ids.shape[1] == self.model.max_sequence_length,
                        f"The input IDs tensor should be truncated to the max_length of {self.model.max_sequence_length}.")

    def test_token_ids_to_embeddings(self):
        token_ids = torch.tensor(self.test_str_token_ids_1, device=self.device).unsqueeze(0)
        embeddings = self.model.token_ids_to_embeddings(token_ids)

        self.assertIsInstance(embeddings, torch.Tensor, "The output should be a torch.Tensor.")
        self.assertEqual(embeddings.device, self.device, "The embeddings are not on the correct device.")

        expected_shape = (1, len(self.test_str_token_ids_1), self.model.embedding_length)
        self.assertEqual(embeddings.shape, expected_shape, "The shape of the embeddings is incorrect.")

        self.assertNotEqual(torch.all(torch.eq(embeddings, torch.tensor(0.0, device=self.device))), True, "The embeddings should not be all zeros.")


    def test_early_embedding(self):
        expected_shape = (1, len(self.test_str_token_ids_1), self.model.embedding_length)
        embeddings = self.model.early_embedding(self.test_str_1)

        self.assertIsInstance(embeddings, torch.Tensor, "The output should be a torch.Tensor.")
        self.assertEqual(embeddings.device, self.device, "The embeddings are not on the correct device.")

        self.assertEqual(embeddings.shape, expected_shape, "The output tensor has an incorrect shape.")
        self.assertTrue(torch.all(embeddings != 0), "The embedding contains unexpected zero elements.")
        self.assertGreater(embeddings.var(), 0, "The embedding variance should be greater than zero.")

    def test_early_embedding_batch(self):
        expected_shape = (2, self.model.max_sequence_length, self.model.embedding_length)
        embeddings = self.model.early_embedding_batch([self.test_str_1, self.test_str_2])

        self.assertIsInstance(embeddings, torch.Tensor, "The output should be a torch.Tensor.")
        self.assertEqual(embeddings.device, self.device, "The embeddings are not on the correct device.")

        self.assertEqual(embeddings.shape, expected_shape, "The output tensor has an incorrect shape.")
        self.assertGreater(embeddings.var(), 0, "The embedding variance should be greater than zero.")

    def test_late_embedding(self):
        expected_shape = (1, 1, self.model.embedding_length)
        embeddings = self.model.late_embedding(self.test_str_1)

        self.assertIsInstance(embeddings, torch.Tensor, "The output should be a torch.Tensor.")
        self.assertEqual(embeddings.device, self.device, "The embeddings are not on the correct device.")

        self.assertEqual(embeddings.shape, expected_shape, "The output tensor has an incorrect shape.")
        self.assertTrue(torch.all(embeddings != 0), "The embedding contains unexpected zero elements.")
        self.assertGreater(embeddings.var(), 0, "The embedding variance should be greater than zero.")


    def test_late_embedding_caching(self):
        self.model._clear_cache()  # Assuming there is a method to clear the cache
        initial_embedding = self.model.late_embedding(self.test_str_1)
        cached_embedding = self.model.late_embedding(self.test_str_1)
        self.assertTrue(torch.allclose(initial_embedding, cached_embedding),
                        "The embedding should be retrieved from the cache.")

        start_time = time.time()
        self.model.late_embedding(self.test_str_1)
        cached_time = time.time() - start_time

        start_time = time.time()
        self.model.late_embedding("This is another input")
        uncached_time = time.time() - start_time

        print("what", uncached_time, cached_time)

        self.assertLess(cached_time, uncached_time,
                        "Retrieving from the cache should be faster than computing a new embedding.")

    def test_late_embedding_batch(self):
        expected_shape = (2, 1, self.model.embedding_length)
        embeddings = self.model.late_embedding_batch([self.test_str_1, self.test_str_2])

        self.assertIsInstance(embeddings, torch.Tensor, "The output should be a torch.Tensor.")
        self.assertEqual(embeddings.device, self.device, "The embeddings are not on the correct device.")

        self.assertEqual(embeddings.shape, expected_shape, "The output tensor has an incorrect shape.")
        self.assertTrue(torch.all(embeddings != 0), "The embedding contains unexpected zero elements.")
        self.assertGreater(embeddings.var(), 0, "The embedding variance should be greater than zero.")

    def test_late_embedding_batch_caching(self):
        self.model._clear_cache()  # Assuming there is a method to clear the cache
        initial_embedding = self.model.late_embedding_batch([self.test_str_1, self.test_str_2])
        cached_embedding = self.model.late_embedding_batch([self.test_str_1, self.test_str_2])
        self.assertTrue(torch.allclose(initial_embedding, cached_embedding),
                        "The embedding should be retrieved from the cache.")

        start_time = time.time()
        self.model.late_embedding_batch([self.test_str_1, self.test_str_2])
        cached_time = time.time() - start_time

        start_time = time.time()
        self.model.late_embedding_batch(["This is another input", "This is another input"])
        uncached_time = time.time() - start_time

        self.assertLess(cached_time, uncached_time,
                        "Retrieving from the cache should be faster than computing a new embedding.")


    def test_predict(self):
        embeddings = self.model.early_embedding(self.test_str_2)
        outputs = self.model.predict(embeddings)
        logits = outputs.logits
        last_logits = logits[:, -1, :]
        predicted_token_id = torch.argmax(last_logits, dim=-1).item()
        predicted_token = self.model.tokenizer.decode(predicted_token_id, skip_special_tokens=False)
        self.assertEqual(self.test_str_2_next_token, predicted_token.strip())

    def test_predict_sequence(self):
        embeddings = self.model.early_embedding(self.test_str_2)
        predicted_seq = self.model.predict_sequence(embeddings, stop_tokens=['.', '\n'])
        self.assertEqual(self.test_str_2_next_sentence, predicted_seq['output_string'].strip())

    def test_predict_top_k(self):
        k = 5
        embeddings = self.model.early_embedding(self.test_str_2)
        prediction_top_k = self.model.predict_top_k(embeddings, target_id=self.target_id, k=k)
        print(prediction_top_k)
        self.assertTrue(prediction_top_k['target_in_top_k'])
        self.assertEqual(len(prediction_top_k['top_k_token_ids']), k, "Incorrect number of tokens")
        self.assertIn(self.target_id, prediction_top_k['top_k_token_ids'], "Prediction not in probable tokens")
        self.assertGreater(prediction_top_k['top_k_probs'][0], 0.5, "Low probability detected")

    def test_predict_top_k_sequence(self):
        embeddings = self.model.early_embedding(self.test_str_2)
        top_k_output = self.model.predict_top_k_sequence(embeddings, source_text=self.test_str_2, target_text=' States of America')
        is_top_k = top_k_output['is_top_k']
        probability = top_k_output['probability']
        self.assertTrue(is_top_k)
        self.assertGreater(probability, 0.1, "Low probability detected")

    def test_decode_input_embedding(self):
        embeddings = self.model.early_embedding(self.test_str_1)
        embeddings = embeddings.squeeze(0)
        decoded =self.model.decode_input_embedding(embeddings)
        self.assertEqual(self.test_str_1, decoded, "Decoded embeddings do not equal input string")

    def tearDown(self):
        self.model._clear_cache()


if __name__ == '__main__':
    unittest.main()