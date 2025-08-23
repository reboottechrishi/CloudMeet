
## GENAI Fundamentals: A Beginner's Guide to the Transformer Architecture and AWS

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

<img width="1864" height="1418" alt="image" src="https://github.com/user-attachments/assets/3016ec9e-2d30-4226-ba4f-de96554b5a3e" />

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
## Building GenAI Apps with AWS Bedrock

You've understand about the power of Generative AI (GenAI) and Large Language Models (LLMs), but how do you actually build an application with them? The answer on AWS is **Amazon Bedrock**.

Think of Bedrock as a central control panel for all the latest and greatest foundation models (FMs). Instead of learning a different API for each model, Bedrock gives you a single, serverless API to access FMs from various providers like **AI21 Labs (Jurassic-2)**, **Anthropic (Claude)**, **Meta (Llama)**, and **Amazon's own Titan models**. This unified approach allows you to build sophisticated GenAI applications without managing any underlying infrastructure.

---

### Understanding the Bedrock API Endpoints

Bedrock uses a split API structure to handle different tasks in the model lifecycle. This is a key concept to understand for building and managing your applications. Bedrock offers four different types of Endpoints

* `bedrock`: This is your **management API**. You use it for tasks you do **before** an application goes live, such as managing your models, deploying them, or fine-tuning them. Think of it as the control plane for your models.
* `bedrock-runtime`: This is the **inference API**. You use this for real-time tasks **after** a model is deployed. This is where your application sends a user's prompt to the model and gets a response back. The `InvokeModel` command is the most common here. For experiences like ChatGPT, where the output appears one word at a time, you'd use the `InvokeModelWithResponseStream` command.
* `bedrock-agent`: Used to manage and deploy your LLM agents and their associated knowledge bases.
* `bedrock-agent-runtime`: Used to perform inference against your deployed agents.

Before you can use any of these models, you must request access via the AWS console. The approval process is quick and lets you try out different models and check their pricing, which is billed directly through AWS.
• Must use with an IAM user (not root), User must have relevant Bedrock permissions like "AmazonBedrockFullAccess", "AmazonBedrockReadOnly"
---

### The Two Paths to Customization: Fine-tuning vs. RAG

When you want to give a foundation model new, private information—like your company's documents or your own specific writing style—you have two main options: **fine-tuning** and **Retrieval-Augmented Generation (RAG)**.

#### Fine-tuning: Teaching the Model New Tricks

**Fine-tuning** is the process of continuing the training of an existing foundation model on your own data. You are literally **changing the model itself** by baking new information directly into its parameters.

* **Use Case:** You want to create a chatbot that has a specific personality or a certain brand voice. You can fine-tune a model on your company's support transcripts to teach it to sound helpful and professional. You are making the model "smarter" for your specific needs.
* **Pros:** Eliminates the need for complex "prompt engineering" to get the desired output. It can also save you money in the long run because you don't need to send the extra context in every single prompt.
* **Cons:** It can be expensive and requires a large, labeled dataset. Any new information requires you to re-tune the entire model, which is a static process.

Bedrock supports fine-tuning for specific models like **Titan, Cohere, and Meta models**. You provide labeled training data (e.g., pairs of prompts and the desired completions) and Bedrock handles the process, creating a "custom model" for you to use like any other.

#### Retrieval-Augmented Generation (RAG): The Overcomplicated Search Engine

**RAG** is a popular alternative to fine-tuning that is faster, cheaper, and more flexible. Instead of training the model on new information, you give it access to an external **knowledge base** to query for answers. The LLM then uses this retrieved information to help answer a user's prompt. 

* **Use Case:** You want to build a Q&A chatbot for your company's internal documents or product manuals. You can load all your documents into a RAG system, and when an employee asks a question, the system finds the most relevant information from your documents and uses that context to generate a precise answer. This is a simple way to deliver "AI search" to your team.
* **Pros:**
    * **Faster and Cheaper:** Updating your knowledge base is as simple as updating a database, with no need for a costly fine-tuning job.
    * **Fewer Hallucinations:** You can guide the LLM to use only the provided context, which significantly reduces the risk of it making things up.
    * **Flexibility:** You are not training the model, so you can update your information with new data in real time.

* **Cons:** RAG is sensitive to the quality of your prompt templates and the relevance of the retrieved data. If you get the wrong context, the model may still provide an incorrect answer.

---

### The Technology Behind RAG: Vector Stores & Embeddings

At the heart of RAG are **vector embeddings** and **vector stores**.

* **Embeddings:** An embedding is a numerical representation of text (or an image, audio, etc.). It's like a point in a multi-dimensional space where similar pieces of text are located close to each other. Bedrock uses a special embedding model (**Cohere** or **Amazon Titan Embeddings**) to create these vectors. 
* **Vector Store:** This is a specialized database that stores these embeddings. When a user asks a question, the query is also converted into an embedding. The vector store then performs a "semantic search" to find the most similar embeddings (i.e., the most relevant chunks of text) in the database.

**Bedrock Knowledge Bases** simplify this entire process. You simply upload your documents (e.g., from an S3 bucket), and Bedrock automatically handles:
1.  **Chunking:** Breaking your documents into smaller pieces.
2.  **Embedding:** Creating a vector for each chunk using your chosen embedding model.
3.  **Storing:** Placing the embeddings in a vector store (like **OpenSearch** or **Aurora**).

---

### Building Intelligent Agents with Bedrock Agents

For more complex applications, Bedrock allows you to build **LLM agents**. An agent is an LLM that is given "tools" to perform actions. It's not just a conversational chatbot; it can take action on your behalf.

* **How it Works:** The LLM is given a description of a set of tools, which are usually **Lambda functions**. When a user's prompt requires a tool, the agent's internal "planning module" figures out which tool to use and what information it needs. It then calls the Lambda function with the required parameters to get the answer.

* **Use Cases:**
    * **Customer Service Bot:** A user asks, "What's the status of my order?" The agent can use a "tool" (a Lambda function) to query your order database, retrieve the status, and provide the answer.
    * **Product Research:** An agent can be given a tool that queries a product API. When a user asks for a product's price, the agent uses the tool to look it up in real time.

You can also use a knowledge base as a "tool" for an agent, a powerful technique called **"Agentic RAG."**

To ensure your applications are safe, Bedrock offers **Guardrails**. This feature allows you to filter prompts and responses for specific words, topics, or personally identifiable information (PII) before the model processes them. It even includes a "Contextual Grounding Check" to help prevent hallucinations by measuring how well the response aligns with the provided context.
[Research Paper-Attention is All You Need](https://arxiv.org/abs/1706.03762)    
[Tokwnizer tool](https://platform.openai.com/tokenizer)  


[Video :Deploy LLMs with SageMaker JumpStart](https://www.youtube.com/watch?v=1-AOLoOiuG4)
