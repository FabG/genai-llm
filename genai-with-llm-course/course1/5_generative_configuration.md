# Gen AI LLM - Course 1
## Part 5 - Generative Configuration and Project Lifecycle

###### Below are some key notes from [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms)

### Generative Configuration
LLMs in playgrounds such as Hugging Face website or on AWS expose *inference parameters* that you can use to influence the way that the model makes the final decision about next-word generation. 
Each model exposes a set of configuration parameters that can influence the model's output during inference. Note that these are different than the training parameters which are learned during training time. Instead, these configuration parameters are invoked at inference time and give you control over things like the maximum number of tokens in the completion, and how creative the output is.

[inference_tokens](../../images/inference_tokens.png)


#### Max new tokens
*Max new tokens* is probably the simplest of these parameters, and you can use it to limit the number of tokens that the model will generate. You can think of this as putting a cap on the number of times the model will go through the selection process. 

Here you can see examples of max new tokens being set to 100, 150, or 200. But note how the length of the completion in the example for 200 is shorter. This is because another stop condition was reached, such as the model predicting and end of sequence token. Remember it's max new tokens, not a hard number of new tokens generated.

[max_new_tokens](../../images/max_new_tokens.png)


#### Greedy and random-weighted sample
The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses. Here you can see a selection of words and their probability score next to them. Although we are only showing four words here, imagine that this is a list that carries on to the complete dictionary. Most large language models by default will operate with so-called *greedy* decoding. 

This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability. This method can work very well for short generation but is susceptible to repeated words or repeated sequences of words.
[greedy_sample](../../images/greedy_sample.png)

If you want to generate text that's more natural, more creative and avoids repeating words, you need to use some other controls. *Random sampling* is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection. 

For example, in the illustration, the word banana has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected. By using this sampling technique, we reduce the likelihood that words will be repeated. However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense. 
[random_weighted_sample](../../images/random_weighted_sample.png)

Note that in some implementations, you may need to disable greedy and enable random sampling explicitly. For example, the Hugging Face transformers implementation that we use in the lab requires that we set do sample to equal true.


#### Top p and Top k sampling
Let's explore top k and top p sampling techniques to help limit the random sampling and increase the chance that the output will be sensible. Two Settings, top p and top k are sampling techniques that we can use to help limit the random sampling and increase the chance that the output will be sensible. 


[top_p_top_k_sampling](../../images/top_p_top_k_sampling.png)

To limit the options while still allowing some variability, you can specify a *top k* value which instructs the model to choose from only the k tokens with the highest probability. 

In this example here, k is set to three, so you're restricting the model to choose from these three options. The model then selects from these options using the probability weighting and in this case, it chooses donut as the next word. This method can help the model have some randomness while preventing the selection of highly improbable completion words. This in turn makes your text generation more likely to sound reasonable and to make sense. 
[top_k_example](../../images/top_k_example.png)



Alternatively, you can use the *top p* setting to limit the random sampling to the predictions whose combined probabilities do not exceed p. For example, if you set p to equal 0.3, the options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3. The model then uses the random probability weighting method to choose from these tokens. With top k, you specify the number of tokens to randomly choose from, and with top p, you specify the total probability that you want the model to choose from. 
[top_p_example](../../images/top_p_example.png)


#### Temperature
In contrast to the top k and top p parameters, changing the *temperature* actually alters the predictions that the model will make. 
[temperature](../../images/temperature.png)


If you choose a low value of temperature, say *less than one*, the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words. You can see this here in the blue bars beside the table, which show a probability bar chart turned on its side. Most of the probability here is concentrated on the word cake. The model will select from this distribution using random sampling and the resulting text will be less random and will more closely follow the most likely word sequences that the model learned during training. 


If instead you set the temperature to a higher value, say, *greater than one*, then the model will calculate a broader flatter probability distribution for the next token. Notice that in contrast to the blue bars, the probability is more evenly spread across the tokens. This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help you generate text that sounds more creative. If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.

[temperature_high_cold](../../temperature_high_cold.png)



[xxxx](../../images/xxxx)
[xxxx](../../images/xxxx)
[xxxx](../../images/xxxx)


### Generative AI project lifecycle
