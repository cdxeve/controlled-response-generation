# controlled-response-generation
Explicitly controlling pattern and content of response generation

### Train 

    dataset=AmazonQA
    
    #forward

    python train.py --dataset ${dataset} --cuda_device 0 --epoch 80 --batch_size 128 --mode forward

    #backward

    python train.py --dataset ${dataset} --cuda_device 0 --epoch 80 --batch_size 128 --mode backward

### Inference 
#### Generate pattern-controlled response

    python test.py --dataset ${dataset} --cuda_device 0 

#### Generate pattern and content controlled response 

    python evaluate.py --dataset ${dataset} --cuda_device 0 --epoch 80 --batch_size 128 ----evaluate_start 0 --evaulate_end 10




