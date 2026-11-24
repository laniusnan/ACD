import json
from pathlib import Path
import tempfile
import unittest

from tools.run_pair_sweep import advance, digest, read, result_for, score, write


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.protocol = read('configs/studies/cifar100_seven_arms_20261118.json')
        self.protocol['pairs'] = ['res56_res20']
        base = read('configs/res56_res20.json')
        base = {k:v for k,v in base.items() if k not in ('evidence','method','etf','seed')}
        base['feature_lift'] = 128
        self.manifest = {'pairs':{'res56_res20':{'base_config':base,'teacher_checkpoint':'teacher.pth'}}}
        self.state = {'jobs':[], 'shortlists':{}, 'selections':{}}
        for i, candidate in enumerate(self.protocol['candidates']):
            config = self.output/f'{candidate}.json'
            write(config,base|self.protocol['candidates'][candidate])
            self.state['jobs'].append({'id':candidate,'pair':'res56_res20','stage':'screen',
                'candidate':candidate,'config':str(config),'config_sha256':digest(config),
                'output':str(self.output/candidate),'status':'completed','score':{'top1':70+i,'ce':1.}})

    def test_promote_then_freeze_one_recipe_for_exactly_seven_arms(self):
        advance(self.output,self.state,self.manifest,self.protocol)
        refined = [j for j in self.state['jobs'] if j['stage']=='refine']
        self.assertEqual(len(refined),2)
        self.assertFalse(self.state['selections'])
        for i,job in enumerate(refined):
            job.update(status='completed',score={'top1':75+i,'ce':.9})
        advance(self.output,self.state,self.manifest,self.protocol)
        finals = [j for j in self.state['jobs'] if j['stage']=='final']
        self.assertEqual(len(finals),7)
        configs = [read(j['config']) for j in finals]
        common = [{k:v for k,v in cfg.items() if k not in ('method','etf','rotating_etf')} for cfg in configs]
        self.assertTrue(all(cfg==common[0] for cfg in common))
        self.assertEqual(common[0]['feature_lift'],128)
        self.assertEqual(common[0]['seed'],3301)
        self.assertEqual(common[0]['validation_per_class'],0)
        self.assertEqual(common[0]['lr'], self.protocol['candidates'][refined[1]['candidate']]['lr'])
        advance(self.output,self.state,self.manifest,self.protocol)
        self.assertEqual(len(self.state['jobs']),13)

    def test_failed_candidate_prevents_silent_recipe_selection(self):
        self.state['jobs'][0]['status']='failed'
        advance(self.output,self.state,self.manifest,self.protocol)
        self.assertEqual(len(self.state['jobs']),4)
        self.assertFalse(self.state['shortlists'])
        self.assertFalse(self.state['selections'])

    def test_screen_ranking_ignores_later_refinement_epochs(self):
        rows=[{'epoch':epoch,'top1':60. if epoch<=60 else 99.,'ce':1.} for epoch in range(1,241)]
        (self.output/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.assertEqual(score(self.output,60)['top1'],60.)
        self.assertEqual(score(self.output,240)['top1'],99.)

    def test_wrong_split_and_corrupt_checkpoint_never_count_as_completed(self):
        (self.output/'latest.pth').write_bytes(b'checkpoint')
        marker={'status':'completed','epoch':240,'checkpoint_sha256':digest(self.output/'latest.pth'),
                'evaluation_split':'official_test','validation':{'samples':10000}}
        write(self.output/'final.json',marker)
        job={'id':'test','stage':'refine','output':str(self.output)}
        with self.assertRaisesRegex(ValueError,'split'):
            result_for(job,self.protocol)
        job['stage']='final'
        self.assertEqual(result_for(job,self.protocol),marker)
        (self.output/'latest.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash'):
            result_for(job,self.protocol)


if __name__=='__main__':
    unittest.main()
