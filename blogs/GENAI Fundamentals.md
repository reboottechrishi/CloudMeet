
### GENAI Fundamentals: A Beginner's Guide to the Transformer Architecture and AWS

In 2023, the world of technology was fundamentally reshaped by a new wave of Generative AI, led by powerful models that could understand and generate human-like text, images, and more. At the heart of this revolution is a single, brilliant innovation: the **Transformer architecture**.

For anyone starting their ML journey or preparing for an exam, understanding this concept is vital. This blog post will break down the Transformer, its core component of self-attention, and how you can apply these models using the AWS ecosystem.

***

### The Transformer: Why Self-Attention is a Game Changer

Before Transformers, models like **Recurrent Neural Networks (RNNs)** and **LSTMs** were the standard for sequential data like language. They processed words one at a time, passing a "hidden state" that represented the meaning so far from word to word. This worked, but it had a critical flaw.

Imagine translating a very long sentence. By the time the RNN gets to the end, the initial words' meaning may have been lost. This created an "information bottleneck" that limited the model's ability to handle long-range dependencies and complex sentences.

• Example - Machine Traslation (ex - english to diffrent language)  
• Encoder / Decoder architecture [ Encoder is to take a piece of text. ex "Please translate me"]. It's going to feed those through a bunch of RNNs sequentially. And between each word,  have this hidden state that represents the meaning up to this point. And we keep passing that along until we get to the end of the sentence that we want to translate. We then pass that sentence over to the decoder, which has the opposite job of taking that hidden state and turning it back into tokens. so if you have long sentance..it's big problem.    

• Encoders and Decoders are RNN’s    
• But, the one vector tying them together creates an information bottleneck   
• RNN’s are sequential in nature, can’t parallelize it : so I'm still processing that encoder one token at a time.
• Information from the start of the sequence may be lost  
  
**So while you can do machine translation using just RNNs and an encoder decoder architecture, it's not the best approach.**    

The big breakthrough came in **2017** with the paper, *"Attention is All You Need."* This paper introduced the **Transformer architecture**, which completely changed the game by getting rid of RNNs. Instead of processing words sequentially, it processes all of them at once **in parallel**. This speed allows for training on massive datasets, which is essential for creating the powerful models we use today.

The magic behind this is a mechanism called **self-attention**.  
Get rid of RNN’s for feed-forward neural networks (FFNN’s)

#### How Self-Attention Works

Each encoder or decoder has a list of embeddings (vectors) for each token that means we are not dealing with with actual words, those words are being tokenized into some sort of numerical representation. Self-attention produces a weighted average of all token embeddings. The magic is in computing the attention weights.  
This results in tokens being tied to other tokens that are important for its context, and a new embedding that captures its “meaning” in context. - If a word has a different meaning in two different contexts, that new resulting token will represent that underlying meaning.
At its core, self-attention allows a model to weigh the importance of every other word in a sentence when it's processing a single word. It creates a weighted average of all the words in the sentence, but the "weights" are the crucial part.

To figure out these weights, the model learns three vectors for each token (a numerical representation of a word or sub-word):

* **Query (Q):** Represents what a word is looking for.
* **Key (K):** Represents what a word has to offer.
* **Value (V):** Represents the actual content of the word.

Every token gets a query (q), key (k), and value (v) vector by multiplying its embedding against these matrices.  
By comparing a word's **Query** to every other word's **Key**, the model calculates an **attention score**. This score determines how much "attention" or importance one word should give to another. A higher score means a stronger relationship. 

This results in a new, context-rich embedding for each word that captures its true meaning within the sentence. For example, in the sentence "The bank is near the river bank," the model can use attention to distinguish between the two meanings of "bank."

To make this even more efficient, **Multi-Headed Self-Attention** allows the model to perform multiple attention calculations in parallel, each focusing on a different aspect of the relationships between words.

***

### Application of Transformers

- Chat    
- Question answering  
- Text classification  (i.e. sentiment analysis-online movie review or a tweet or something)  
- Named entity recognition  
- Summarization  
- Translationn  
- Code generation  
- Text generation  

    
***  

### Generative Pre-Trained transformers:How they work (From Transformers to LLMs/GPT)

The **Generative Pre-trained Transformer (GPT)** models, such as those from OpenAI, are a specific type of LLM built entirely on the **decoder-only** Transformer architecture.

This design is what makes them so good at generating text. Unlike models with both encoders and decoders, GPT models don't have a separate input and output concept. All they do is generate the next token, over and over again. You "prompt" it with a question or command, and it uses a **masked self-attention** layer to prevent it from "peeking" at future words it hasn't generated yet. It then generates the next token based on all the previous tokens in the sequence.

The ability to train on vast amounts of unlabeled text is a key innovation. Instead of being trained for a specific task, the model simply learns to predict the next word in a sequence, effectively "learning language." This is what makes it so versatile.

**GPT architecture**

- Generative Pre-Trained Transformer (ex-GPT-2)
    - Other LLM’s are similar
- **Decoder-only** – stacks of decoder blocks
    - Each consisting of a masked **self-attention layer**, and a **feed-forward neural network**
    - As an aside, BERT consists only of encoders. T5 is an example of a model that uses both encoders and decoders.
- No concept of input, all it does is generate the next token over and over
    - Using attention to maintain relationships to previous words / tokens
    - You “prompt” it with the tokens of your question or whatever
    - It then keeps on generating given the previous tokens
- Getting rid of the idea of inputs and outputs is what allows us to train it on unlabeled piles of text
    - It’s “learning a language” rather than optimizing for some spsecific task
- Hundreds of billions of parameters

#### LLM  key terms & Controls(token, embedding,temp etc.)

* **Tokens:** Numerical representations of words or parts of words. (token coverted into embedding)
* **Embeddings:** mathematical representations (vectors) that encode the “meaning” of a token. A vector that encodes the semantic "meaning" of a token.
  - Top P – Threshold probability for token inclusion (higher = more random)
  - Top K – Alternate mechanism where K candidates exist for token inclusion (higher = more random)
* **Context Window:** The maximum number of tokens an LLM can process at once.
* **Temperature:** A control that adds randomness to the model's output. A low temperature produces more consistent and predictable responses, while a high temperature leads to more creative and varied results.
* **Max tokens –** Limit for total number of tokens (on input or output)

### AWS Foundation Models & SageMaker JumpStart

We talked in general terms how transformers work and specifically how GPT works. But how does this all fit into AWS. So AWS offers something called **foundation models(FM).** So, how do these powerful models fit into the AWS ecosystem?      

The giant, pre-trained transformer models(like GPT, BERT, Claude, LLaMa...) we are fine tuning for specific tasks, or applying to new applications.    
• GPT-n (OpenAI, Microsoft)    
• Claude (Anthropic)    
• BERT (Google)    
• DALL-E (OpenAI, Microsoft)    
• LLaMa (Meta)    
• Segment Anything (Meta)    
• **Where’s Amazon?**    

AWS provides access to these massive, pre-trained models, known as **Foundation Models (FMs)**. Instead of building your own model from scratch, you can use these FMs as a base for your applications. AWS offers FMs from various providers, including:      

- Jurassic-2 (AI21labs)  
    - Multilingual LLMs for text generation  
    - Spanish, French, German, Portuguese, Italian, Dutch  
- Claude (Anthropic)  
    - LLM’s for conversations  
    - Question answering  
    - Workflow automation  
- Stable Diffusion (stability.ai)  
    - Image, art, logo, design generation  
- Llama (Meta)  
    - LLM  
- Amazon Titan  
    - Text summarization  
    - Text generation  
    - Q&A  
    - Embeddings  
        - Personalization  
        - Search  
- AWS Newest model is called **Nova**  

The best way for a beginner to get hands-on with these models is with **Amazon SageMaker JumpStart**. This feature within SageMaker Studio is a one-click hub that lets you deploy and use these FMs right away. You can quickly launch a notebook with a model loaded, ready for you to experiment with, fine-tune with your own data, and deploy to production.

This makes the power of Generative AI and the Transformer architecture accessible to everyone, from students to experienced ML engineers.

---

[Research Paper-Attention is All You Need](https://arxiv.org/abs/1706.03762)    
[Tokwnizer tool](https://platform.openai.com/tokenizer)  


[Video :Deploy LLMs with SageMaker JumpStart](https://www.youtube.com/watch?v=1-AOLoOiuG4)
