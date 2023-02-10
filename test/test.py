import unittest
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import gc
import BERT_Family as BF

testdata = load_dataset('glue', "mrpc", split="validation")
myBF = BF.BFClassification(pretrained_model = "albert-base-v2")
myBF.load_model()
type(myBF.model)

dl = myBF.set_dataset(pd.DataFrame(testdata)[["sentence1", "sentence2"]], 
                                               testdata["label"],
                                               data_type="validation")

s = next(iter(dl))
list(s[0].keys()) == ['input_ids', 'token_type_ids', 'attention_mask']
len(s[0])
len(s[1])

s[0]["input_ids"].shape == (128, 1, 100)

tmp = BF.infinite_iter(dl)
s = next(tmp)
len(s[1])
myBF.max_length
dl.batch_size

s[0]["input_ids"].shape
len(1)



class test_BFClassification(unittest.TestCase):
    def setUp(self):
        self.testdata = load_dataset('glue', "mrpc", split="validation")
        self.myBF = BF.BFClassification()


    def tearDown(self):
        self.testdata = None
        self.myBF = None


    def test_set_dataset(self):
        testDataLoader = self.myBF.set_dataset(pd.DataFrame(self.testdata)[["sentence1", "sentence2"]], 
                                               testdata["label"],
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



        

    



gc.collect()

testdata = load_dataset('glue', "mrpc", split="validation")


tmp = BF.BFClassification()
d = tmp.set_dataset(pd.DataFrame(testdata), data_type="unittest")
d
DataLoader()



def add(a, b):
    return a

class AddTestCase(unittest.TestCase):
    def setUp(self):
        self.args = (100, 99)

    def tearDown(self):
        self.args = None

    def test_add(self):
        expected = 199
        result = add(*self.args)
        #self.assertEqual(expected, result)
        self.assertIsInstance(result, float)


suite = unittest.TestSuite()
suite.addTest(AddTestCase('test_add'))


unittest.TextTestRunner(verbosity=2).run(suite)
