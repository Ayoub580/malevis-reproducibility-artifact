"""Check weight-compatible inputs and the development-only/evaluation boundary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import tensorflow as tf
import original_training_reference as reference
import train_resnet_preprocess_corrected as run

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)


class PreprocessCorrectionTest(unittest.TestCase):
    def test_transforms_and_development_only_resume(self):
        def tiny_model(n,size):
            inp=tf.keras.Input(shape=(*size,3))
            x=tf.keras.layers.Conv2D(4,3,strides=8)(inp)
            x=tf.keras.layers.BatchNormalization()(x)
            base=tf.keras.Model(inp,x);base.trainable=False
            y=tf.keras.layers.GlobalAveragePooling2D()(base.output)
            y=tf.keras.layers.Dense(n,activation='softmax')(y)
            return tf.keras.Model(base.input,y),base
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows={}
            for part in ('fitting','development','evaluation'):
                rows[part]=[]
                for c in ('a','b'):
                    for i in range(2):
                        sid=f'{part}/{c}/{i}.png';p=root/'data'/sid;p.parent.mkdir(parents=True,exist_ok=True)
                        Image.new('RGB',(300,300),(10,80,150)).save(p)
                        rows[part].append(dict(sample_id=sid,class_label=c,partition=part))
            args=SimpleNamespace(output=root/'out',dataset=root/'data',model='resnet50',batch_size=2,
                                 epochs_phase1=1,epochs_phase2=1,custom_epochs=1,restart_incomplete=False,evaluate=False)
            gen=run.make_generators(reference,args.dataset,rows['fitting'],rows['development'],rows['evaluation'],
                                    ['a','b'],2,'resnet50')
            for g in gen:
                self.assertIsNone(g.image_data_generator.rescale)
                self.assertEqual(g.interpolation,'bicubic')
                x,y=g[0]
                np.testing.assert_allclose(x[0,0,0],np.array([150-103.939,80-116.779,10-123.68]),atol=1e-5)
            with patch.dict(reference.MODELS,{'resnet50':tiny_model}):
                result=run.run_one(args,{'classes':['a','b']},rows['fitting'],rows['development'],rows['evaluation'],
                                   'full',42,reference,tf)
                dest=args.output/'resnet50/full/seed_42'
                self.assertEqual(result['status'],'development_complete')
                self.assertFalse((dest/'metrics.json').exists())
                self.assertFalse((dest/'predictions.csv').exists())
                self.assertTrue((dest/'development_per_class.csv').exists())
                self.assertTrue((dest/'selected.keras').exists())
                args.evaluate=True
                with patch.object(run,'train_phases',side_effect=AssertionError('Must reuse checkpoint')):
                    result=run.run_one(args,{'classes':['a','b']},rows['fitting'],rows['development'],rows['evaluation'],
                                       'full',42,reference,tf)
                self.assertEqual(result['evaluation_images'],4)
                self.assertTrue((dest/'metrics.json').exists())


if __name__=='__main__':
    unittest.main()
