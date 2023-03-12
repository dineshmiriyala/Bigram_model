# Bigram_model:
In this repo I will use reddit conversation to define a Bi gram model that will predict sentences based on the conversational data that is provided.
It is not the best approach for this problem but will give us an idea of how NLP works.
The data set is from https://affect.media.mit.edu/neural_chat/datasets/.

I have also created a text preprocessing python file that downloads the data from url and writes into reddit_convos.txt file.

In this repo I will be first using lookup table to generate sequence based on the dataset. Later on I will be training a simple neural network to do the same thing.
For optimized Bi-gram lookup table I got a loss of : **2.3944**.

My prediction is that the neural network will also have similar loss values to that of lookup approach. 

This is because we are only using character level prediction, and it can only get so optimized. No matter how complicated the neural network is it won't be more efficient than the lookup table. 
In fact, the predicted sequence will also be similar to each other.

# Loss Function:
The loss function we used here is **average negative log likelihood**. In general the lower the loss value, the better the prediction.

It is called as **Cross Entropy**. Refer to: https://en.wikipedia.org/wiki/Cross_entropy

To find the loss function value:

1. We will find the probability matrix for lookup table(lookup-table / sum(lookup-table)).

2. We will then apply log function to particular char sequence and add the values. we are using
    log values because it is a monotonically increasing function which will make values easier to handle,
    and we can add these probabilities. because the probability of two sequences happening one after other is derived by
    multiplying those two values. since log multiplication is equal to summing of values.

3. We will then apply negative sign to log value.

4. We will then take the average value of negative log value as our main loss value.

# Sample prediction from Bi gram lookup table:
    Habond tstsery hasis gatis prenc wofan Ivirer pinidorel sutskl tyo roueaburan I hevend are g.
    Douped.
    Tred.
    Ne meregous an ghe ngnt on amount Imovero wndclk ureay th.
    I fe wof ly bld.
    I alatingi sh an walpe d ayblitytrea veret ice wasot.
    Theoforalenalara buffffoleore.
    I ng thatho.
    I y IPan ami f eabatoreinin toim aye s mas e r d s top jungl ug ind spane y ll.
    Cr Ivom d mpalinn t.


All the files can be found in this repo. There is only the processed dataset, if you want the original file refer to https://affect.media.mit.edu/neural_chat/datasets/.

# Neural Network:
**Since the network save files are huge to upload, you need to train first and then use the model. All the files would be saved in the Bigram_model directory.**

**There are no further optimizations since it does not make sense to optimize a network that only has one context character, because no matter how low the loss is the prediction is still gibberish.**

After the training is done, you can load the model and generate text from it. Like we anticipated the network will have a loss of *0.23944* in best case scenario, and it is impossible to go lower than that, believe me I tried. 
Optimizing it further is inefficient and impractical as no matter the loss the context for prediction is only one character, so it would be gibberish. 

The idea is that: we will be using a simple neural network that has only one layer and a softmax as output layer. We will be calculating cross entropy(refer to loss section) and will be passing a backward pass to calculate gradients 
for each variable and will be updating the gradients in the direction of lower loss. 

The best loss I could be able to achieve is *2.651* that to after training it over 100 iterations through the whole dataset. 

# Sample prediction from simple neural network:
    I araikoutsQpy f shg fRpll ouorG wqLSVL yoreo ondd unee syvthk tyo rjuOVFqhan HTRivenXTDgWDPAEppe i.
    fule nUNd s mqus an fhakelnvand XUlont Etovaso woefin ureHzese.
    OBNakuve it anN.
    I akDqmejk.
    be.
    FbBy fomRLQCzfedszzveGCusomwDTEQHwaren t mg sie he f od avedFZs flin ingWRve tho.
    I y Cd m WTh e ccK tinedThe tlWQCDxike mHicaLlMMUleson isQYl uSBTRWFlo he y io XwJ.
    xzS cScoEPYRn t.
    Gcnust thqwery.
    EDs be yd he alyEmc.

by comparing those two outputs we can tell that both of those are trash and need bigger models with more context to get more desired results. 

I will be using the same data set to build MLP model (multi-layer perceptron) to predict more useful results. There I will be using words instead of characters and I will be having more than one context.

This project is done with production ready code.