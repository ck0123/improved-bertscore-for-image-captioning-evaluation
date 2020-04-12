## Improved BERTScore for image captioning evaluation
Implementation of paper: Improving Image Captioning Evaluation by Considering Inter References Variance (ACL2020)



## Usage:
Recently, this repo provides two metrics ('with BERT' and 'simple')

* python3 run_metric.py    
    
* python3 run_metric_simple.py 

## example data:

example/example.json (you can modify this file for your own datasets)   



Fields explanation:  
* "refs": reference captions (each sample 5 references)    
* "cand": candidate caption (each sample 1 candidate)
* "refs_hid": contextual embeddings of reference captions
* "cand_hid": contextual embeddings of cand captions
* "mismatch": mismatches marks computed from all of reference captions
* "metric_result": scores on our metric  

  
NOTE:   
we also provide Flickr 8K Expert Annotation file with our format 'example/flickr.json'  
you can easily reproduce our result following run_metric.py lines 223-235.


## Dependencies:
pytorch-pretrained-bert==0.6.2 (old version of [huggingface/transformers](https://github.com/huggingface/transformers))     
torch==0.4.1  
bert_score==0.1.2 (already in this repo)


