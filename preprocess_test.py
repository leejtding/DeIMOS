import unittest
import sample_images
import preprocess
import os
import tensorflow as tf

class TestPreprocess(unittest.TestCase):
    def test_preprocess_20(self):
        location = os.path.join('data', 'unit_test')
        sample_images.sample_images(location, seed=123, sample_num=20)
        labelled, unlabelled = preprocess.get_data('data/unit_test/hirise-map-proj-v3_2')
        self.assertTrue(3 == len(list(labelled)))
        self.assertTrue(17 == len(list(unlabelled)))

        for img in labelled:
            print(img)
            self.assertEqual(img[0].shape, (227, 227, 1))
            self.assertEqual(img[0].dtype, tf.float32)
            self.assertEqual(img[1].shape, ())
            self.assertEqual(img[1].dtype, tf.int32)

        for img in unlabelled:
            self.assertEqual(img.shape, (227, 227, 1))
            self.assertEqual(img.dtype, tf.float32)

    def test_preprocess_1000(self):
        location = os.path.join('data', 'unit_test')
        sample_images.sample_images(location, seed=123, sample_num=1000)
        labelled, unlabelled = preprocess.get_data('data/unit_test/hirise-map-proj-v3_2')
        self.assertTrue(198 == len(list(labelled)))
        self.assertTrue(802 == len(list(unlabelled)))

if __name__ == '__main__':
    unittest.main()
