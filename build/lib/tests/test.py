import unittest
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import BERT_Family as BF

class test_BFClassification(unittest.TestCase):
    def setUp(self):
        self.testdata = load_dataset('glue', "mrpc", split="validation")
        self.myBF = BF.BFClassification()


    def tearDown(self):
        self.testdata = None
        self.myBF = None


    def test_set_dataset(self):
        testDataLoader = self.myBF.set_dataset(pd.DataFrame(self.testdata)[["sentence1", "sentence2"]], 
                                               self.testdata["label"],
                                               data_type="validation")
        self.assertIsInstance(testDataLoader, DataLoader)

        testModelInputType = next(iter(testDataLoader))
        #test output with x and y (len = 2)
        self.assertEqual(len(testModelInputType), 2)
        self.assertEqual(len(testModelInputType[0]), 3)
        self.assertEqual(list(testModelInputType[0].keys()), ['input_ids', 'token_type_ids', 'attention_mask'])
        self.assertEqual(testModelInputType[0]["input_ids"].shape, (testDataLoader.batch_size, 1, self.myBF.max_length))
        self.assertEqual(len(testModelInputType[1]), testDataLoader.batch_size)


    def test_load_model(self):
        self.myBF.load_model()
        self.assertIsNotNone(self.myBF.model)

        model_name = type(self.myBF.model).__name__
        result = str.find(str.lower(model_name), "bert")
        self.assertNotEqual(result, -1)

if __name__=='__main__':
    unittest.main()

