# dataloader fails
- Fails because collator doesn't work
# `ppo_trainer.generate` fails
- Input must be `Tensor` not `list`
# KeyError "prompts"
- Need to include prompts field in dataset
# Wrong device
- `prefix_tensors` is on cuda but `batch["prompt"]` is on cpu
    - Why?
        - When `ppo_trainer` generates tokens, those tokens are on gpu because the 
            policy model is on GPU. 
        - `batch["prompt"]` comes from the dataloader. It is not being put on the GPU
            right now. However, this problem is not ocurring during training only 
            during testing. What is the difference?
        - The difference is the training dataset is passed into `ppo_trainer`.
            This automatically puts the input on cuda. I need to do that manually during
            testing.
# KeyError `target`
    - The target is not being included in the batch
# Evaluation variance
    - Sampling decoding will result in some variance in the output. 
    - I can't find any information in `zhang2023rmt` about this so I am going to continue using my greedy decoding approach
# IndexError
    - `torch.cat((prefix_tensors[i], batch["prompt"][i]))`
    - Either prefix_tensors or batch["prompt"] is out of range
    - for i in range(len(prefix_tensors))
        - Changed i to loop over length of prefix tensors. 
        - This should handle when batches are not same size
#  ValueError: maximum supported dimension for an ndarray is 32, found 5000
    - mean_reward = np.ndarray(total_rewards).mean(0)
    - Something strange is happening to total rewards
# Reward model
- The reward model is a sentiment classifier
    - DistilBertForSequenceClassification
    - Default `sentiment-analysis` pipeline on huggingface 
# Test model additions
- Test both positive and negative targets
- Test with positive, neutral, and negative prompts
- Let me refactor this method so the actual testing receives a dataset
    - The dataset is dependent on the target
    - The outer loop will be the positive, neutral, and negative or the number of prompt datasets
    - The inner loop will be positive or negative, the target attribute
- There will be a single table for all of these called `test_table`
    - I need to include the dataset name in this table
- I might need to adjust generation length
    - It is set to 20

# Model is not being saved
- For some reason the model is not being saved after training
    - Fixed. Was always being 

# Rewards is a string?
best_reward_index = torch.tensor(rewards).argmax()
TypeError: new(): invalid data type 'str'
- `dataset` argument should have been last not first.

# Test logging
- Missing positive, negative, and neutral bar graphs
    - Messed up dictionary
- Test prompt needs to be decoded
- Test table only 80 rows?
    - Should be all examples
    - `log_test_stats` gets the best from the batch (256)
- saved model has wrong name
    - `wandb.run.name` works 
# Dictionary not being logged as bar graph
- Results in line plots with a single point

# Failed generation
- At some point during training, generation fails.
- I am not sure why it fails
- I am not sure this is even the true problem
- I realized that I still had a batch size of 1
    - Is this going to fix my problem?
- It is running now.
    Is there anything I can do while it is running?
    - This might have fixed the issue, we will see.

# Swift sky problems
- Model not saved
    - Debugging this next
- Test total table only for positive dataset
    - Don't need to test on same target
        - Its ok
    - Should write results to csv
        - Added this
    - Wrong indentation
# Comic universe
- Why is the negative -> positive test so bad?
    - Is there a bug?
    - Why is negative -> negative so bad?
- What is the right response model?
    - Does the paper say?
    