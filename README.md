# Bigram_model:
In this repo I will use reddit conversation to define a Bi gram model that will predict sentences based on the conversational data that is provided.
It is not the best approach for this problem but will give us an idea of how NLP works.
The data set is from https://affect.media.mit.edu/neural_chat/datasets/.

In this repo I will be first using lookup table to generate sequence based on the dataset. Later on I will be training a simple neural network to do the same thing.
For optimized Bi-gram lookup table I got a loss of :**2.3944**.

My prediction is that the neural network will also have similar loss values to that of lookup approach. 

This is because we are only using character level prediction, and it can only get so optimized. No matter how complicated the neural network is it won't be more efficient than the lookup table. 
In fact, the predicted sequence will also be similar to each other.

# Loss Function:
The loss function we used here is **average negative log likelihood**. In general the lower the loss value, the better the prediction.

To find the loss function value:

1. We will find the probability matrix for lookup table(lookup-table / sum(lookup-table)).

2. We will then apply log function to particular char sequence and add the values. we are using
    log values because it is a monotonically increasing function which will make values easier to handle,
    and we can add these probabilities. because the probability of two sequences happening one after other is derived by
    multiplying those two values. since log multiplication is equal to summing of values.

3. We will then apply negative sign to log value.

4. We will then take the average value of negative log value as our main loss value.

# Sample prediction from Bi gram lookup table:
    W trers peetoche byoreer ovese n iniventatll ndnsance t yotor odo Th er Heous wst tho hendd om t id fris y te menes r n eraysakfonkevi ca ake stlate he cavern pe morereen foficans ticie t.
    OOred droutawhasey utt th fatorifilath t inerlady donge juldo bust ave.
    I achaloint tireal ps thiktsthot c ik c.
    S ent r blys br chic.
    If mistod pal Extanlivednswoneakenthe ory as ong.
    Trs ald to ve wot.
    Yo t at it isundete tintes.
    Pil wn acase yof an ifares we st banger erenteve ptathan here se e s wiop cthofoutysthent bolir g.
    Anesoraren cortsushind ayo LAMyonexinodse ar it ckethin blinaf an med Cancouce atio owhe ikinor itharous I mmerird ren apeells.
    BGape I It an areh jut t isken m we ithary n t ourone tt.

All the files can be found in this repo. There is only the processed dataset, if you want the original file refer to https://affect.media.mit.edu/neural_chat/datasets/.