# AI Solutions on Azure

## What is AI?
Artificial Intelligence (AI) enables software to perform tasks that appear to require human cognition: understanding input (text, speech, images, video), reasoning over data, and generating or selecting appropriate outputs.

Modern AI solutions are typically built on machine learning and foundation models that learn patterns and semantic relationships from large datasets.

## Common AI Capabilities

| Capability | Description | Example Use Case |
|-----------|-------------|------------------|
| Generative AI | Creates original content from natural language prompts. | Auto-generate property listing descriptions. |
| Agents | Autonomously decide and perform multi-step tasks. | Calendar assistant arranging transport. |
| Computer Vision | Interprets images and video streams. | Identifies items at automated retail checkout. |
| Speech | Speech-to-text and text-to-speech. | Voice-driven Q&A in an app. |
| Natural Language Processing | Analyzes, summarizes, classifies, translates text. | Sentiment analysis of social posts. |
| Information Extraction | Pulls structured data from documents, images, audio. | Extracts totals and line items from receipts. |
| Decision Support | Predicts outcomes from historical patterns. | Forecasts real estate pricing trends. |

## Azure AI Services
Azure provides managed APIs and model hosting so you can add AI features without building everything from scratch.

| Service | Core Purpose | Typical Scenarios |
|---------|--------------|-------------------|
| Azure OpenAI | Access to GPT family, fine-tuning, embeddings, image generation (DALL·E). | Chat, content generation, code assist, RAG. |
| Azure AI Vision | General image/video analysis (objects, captions, OCR). | Image tagging, content moderation, retail vision. |
| Azure AI Speech | Speech-to-text, text-to-speech, translation, speaker features. | Voice assistants, call analytics, accessibility. |
| Azure AI Language | Text analytics, entities, sentiment, summarization, Q&A, conversational language. | Knowledge bots, feedback mining, document summarization. |
| Azure AI Content Safety | Detects unsafe / policy-sensitive text & images. | Moderating user-generated content. |
| Azure AI Translator | Multi-language text translation. | Real-time localization, multilingual chat. |
| Azure AI Face | Face detection & analysis (restricted advanced features). | Access control, anonymized analytics. |
| Azure AI Custom Vision | Train custom image classifiers / object detectors. | Domain-specific product or defect recognition. |
| Azure AI Document Intelligence | Structured field extraction from forms & documents. | Invoice/receipt automation, form ingestion. |
| Azure AI Content Understanding | Multimodal extraction across documents, images, audio, video. | Unified content pipelines, governance. |
| Azure AI Search | Cognitive & vector search over content using skillsets + semantic ranking. | RAG grounding, enterprise search portals. |

## Azure AI Foundry
Azure AI Foundry is a unified platform for building, evaluating, deploying, and managing AI solutions on Azure.


## Responsible AI
Building AI requires mitigating risk and promoting beneficial impact.

### Core Principles
| Principle | Goal | Key Engineering Actions |
|-----------|------|--------------------------|
| Fairness | Avoid unjust bias / disparate impact | Curate representative data; slice metrics by demographic segments; bias mitigation techniques |
| Reliability & Safety | Consistent, bounded behavior | Robust testing, fallback logic, confidence thresholds, red-team scenarios |
| Privacy & Security | Protect user data & content | Data minimization, encryption at rest/in transit, differential access, secure secret management |
| Inclusiveness | Benefit all users | Accessible UX, diverse user testing, language and modality support |
| Transparency | Explain system purpose & limits | Disclose model use, surface confidence, document data lineage & evaluation results |
| Accountability | Human responsibility & governance | Clear ownership, audit trails, incident response runbooks, policy & compliance reviews |

## Choose and deploy models from the model catalog in Azure AI Foundry portal

### Focus on a modality, task, or tool
Language models like GPT-4 and Mistral Large are also known as chat completion models, designed to generate coherent and contextually appropriate text-based responses. When you need higher levels of performance in complex tasks like math, coding, science, strategy, and logistics, you can also use reasoning models like DeepSeek-R1 and o1.

Beyond text-based AI, some models are multi-modal, meaning they can process images, audio, and other data types alongside text. Models like GPT-4o and Phi3-vision are capable of analyzing and generating both text and images. Multi-modal models are useful when your application needs to process and understand images, such as in computer vision or document analysis. Or when you want to build an AI app that interacts with visual content, such as a digital tutor explaining images or charts.

If your use case involves generating images, tools like DALL·E 3 and Stability AI can create realistic visuals from text prompts. Image generation models are great for designing marketing materials, illustrations, or digital art.

### How do I select the best model for my use case?
To select the best language model for your use case, you need to decide on what criteria you're using to filter the models. The criteria are the necessary characteristics you identify for a model. Four characteristics you can consider are:

- **Task type**: What type of task do you need the model to perform? Does it include the understanding of only text, or also audio, or video, or multiple modalities?
- **Precision**: Is the base model good enough or do you need a fine-tuned model that is trained on a specific skill or dataset?
- **Openness**: Do you want to be able to fine-tune the model yourself?
- **Deployment**: Do you want to deploy the model locally, on a serverless endpoint, or do you want to manage the deployment infrastructure?

### Optimize model performance
After you deploy your model to an endpoint, you can start interacting with it to see how it works. Let's explore how you can use prompt engineering techniques to optimize your model's performance.

#### Ask for better question suggestions
When you want the model to help refine your question, you can ask it to suggest better ways to phrase your query. You can also ask it to provide extra context.

When you encourage the model to suggest clarifications or extra details, it can help you ask the right questions and achieve better, more accurate answers in fewer interactions.

#### Specify the desired format for responses
When you want the model to generate output in a specific format, you can provide a template or structure in your prompt.

For example, if you're a sports reporting composing a historical article, you can request that the model follow a specific template, which includes headings, bullet points, and data breakdowns.

| | No template specified | With template specified |
|---|---|---|
| System prompt | You're a helpful AI assistant. | You're a helpful AI assistant for sports reporters. |
| User prompt | What happened in the 2018 Football World Cup final? | What happened in the 2018 Football World Cup final? Format the result to show the match date, location, and the two teams competing. Then the final score, and finally any notable events that occurred during the match. |
| Model response | The 2018 FIFA World Cup final took place on July 15, 2018, at the Luzhniki Stadium in Moscow, Russia. France faced Croatia in the match. France emerged victorious with a 4-2 win. The goals for France were scored by Mario Mandzukic (own goal), Antoine Griezmann (penalty), Paul Pogba, and Kylian Mbappé. Croatia's goals were scored by Ivan Perišić and Mandzukic. Kylian Mbappé, at just 19 years old, became the second youngest player to score in a World Cup final, after Pelé. This victory marked France's second World Cup title, their first being in 1998. | Match Date: July 15, 2018<br>Location: Luzhniki Stadium, Moscow, Russia<br>Teams Competing: France vs. Croatia<br>Final Score: France 4, Croatia 2<br><br>Notable Events:<br>First Half: France took an early lead in the 18th minute with an own goal by Mario Mandžukić after a cross from Antoine Griezmann. Croatia equalized in the 28th minute with a header from Ivan Perišić, who had just come on as a substitute. Just before halftime, in the 44th minute, Paul Pogba scored from a free kick to put France back in the lead.<br>Second Half: The game saw a dramatic increase in intensity after the break. In the 68th minute, Kylian Mbappé scored his first World Cup goal, bringing the score to 3-1 in favor of France. Croatia managed to score again in the 69th minute when Mandžukić headed in a cross from Perišić, reducing the deficit to 3-2.<br>Final Moments: In the 81st minute, a controversial penalty was awarded to France after a handball by Croatian defender Dejan Lovren. Griezmann converted the penalty, sealing the victory for France with a 4-2 scoreline.<br>Additional Context: This match was notable for being the youngest ever World Cup final in terms of average age, featuring players like Mbappé and N'Golo Kanté. France's victory marked their second World Cup title, having previously won in 1998. Croatia reached the final for the first time in their history, finishing as runners-up. |

#### Add context
When you want the model to focus on specific topics, you can specify the context to consider. You can also tell the model to ignore irrelevant information.

For example, if you're planning a trip, you can provide the model with more context to help improve the relevance of its response.

#### Apply model optimization strategies

Prompt engineering can be an effective way to optimize model responses, but in some cases it may not provide sufficient context or guidance to always meet your exact needs. As a developer, you can consider the following additional optimization strategies to improve the relevance of your generative AI application's responses:

- **Retrieval Augmented Generation (RAG)**: A technique that involves using a data source to provide grounding context to prompts. RAG can be a useful approach when you need the model to answer questions based on a specific knowledge domain or when you need the model to consider information related to events that occurred after the training data on which the model is based.

- **Fine-tuning**: A technique that involves extending the training of a foundation model by providing example prompts and responses that reflect the desired output format and style.

### Hands On Lab
**Choose and deploy a language model:**  
https://microsoftlearning.github.io/mslearn-ai-studio/Instructions/02-Explore-model-catalog.html

## Get started with prompt flow to develop language model apps in the Azure AI Foundry

### Understand the development lifecycle of a large language model (LLM) app
Before understanding how to work with prompt flow, let's explore the development lifecycle of a Large Language Model (LLM) application.

The lifecycle consists of the following stages:

1. **Initialization**: Define the use case and design the solution.
2. **Experimentation**: Develop a flow and test with a small dataset.
3. **Evaluation and refinement**: Assess the flow with a larger dataset.
4. **Production**: Deploy and monitor the flow and application.
During both evaluation and refinement, and production, you might find that your solution needs to be improved. You can revert back to experimentation during which you develop your flow continuously, until you're satisfied with the results.

Let's explore each of these phases in more detail.

#### Initialization
Imagine you want to design and develop an LLM application to classify news articles. Before you start creating anything, you need to define what categories you want as output. You need to understand what a typical news article looks like, how you present the article as input to your application, and how the application generates the desired output.

In other words, during initialization you:

1. **Define the objective**
2. **Collect a sample dataset**
3. **Build a basic prompt**
4. **Design the flow**
To design, develop, and test an LLM application, you need a sample dataset that serves as the input. A sample dataset is a small representative subset of the data you eventually expect to parse as input to your LLM application.

When collecting or creating the sample dataset, you should ensure diversity in the data to cover various scenarios and edge cases. You should also remove any privacy sensitive information from the dataset to avoid any vulnerabilities.

#### Experimentation
You collected a sample dataset of news articles, and decided on which categories you want the articles to be classified into. You designed a flow that takes a news article as input, and uses an LLM to classify the article. To test whether your flow generates the expected output, you run it against your sample dataset.

The experimentation phase is an iterative process during which you:

1. **Run the flow** against a sample dataset
2. **Evaluate the prompt's performance**
3. **Check satisfaction** with the result - if satisfied, move on to evaluation and refinement
4. **Modify the flow** by changing the prompt or flow itself if improvement is needed

#### Evaluation and refinement
When you're satisfied with the output of the flow that classifies news articles, based on the sample dataset, you can assess the flow's performance against a larger dataset.

By testing the flow on a larger dataset, you can evaluate how well the LLM application generalizes to new data. During evaluation, you can identify potential bottlenecks or areas for optimization or refinement.

When you edit your flow, you should first run it against a smaller dataset before running it again against a larger dataset. Testing your flow with a smaller dataset allows you to more quickly respond to any issues.

Once your LLM application appears to be robust and reliable in handling various scenarios, you can decide to move the LLM application to production.

#### Production
Finally, your news article classification application is ready for production.

During production, you:

1. **Optimize the flow** that classifies incoming articles for efficiency and effectiveness
2. **Deploy your flow** to an endpoint. When you call the endpoint, the flow is triggered to run and the desired output is generated
3. **Monitor the performance** of your solution by collecting usage data and end-user feedback. By understanding how the application performs, you can improve the flow whenever necessary

### Hands On Lab
**Use a prompt flow to manage conversation in a chat app:**  
https://microsoftlearning.github.io/mslearn-ai-studio/Instructions/03-Use-prompt-flow-chat.html

## Develop a RAG-based solution with your own data using Azure AI Foundry
Language models are growing in popularity as they create impressive coherent answers to a user's questions. Especially when a user interacts with a language model through chat, it provides an intuitive way to get the information they need.

One prevalent challenge when implementing language models through chat is the so-called groundedness, which refers to whether a response is rooted, connected, or anchored in reality or a specific context. In other words, groundedness refers to whether the response of a language model is based on factual information.

### Ungrounded prompts and responses
When you use a language model to generate a response to a prompt, the only information that the model has to base the answer on comes from the data on which it was trained - which is often just a large volume of uncontextualized text from the Internet or some other source.

The result will likely be a grammatically coherent and logical response to the prompt, but because it isn't grounded in relevant, factual data, it's uncontextualized; and may in fact be inaccurate and include "invented" information.
### Grounded prompts and responses
In contrast, you can use a data source to ground the prompt with some relevant, factual context. The prompt can then be submitted to a language model, including the grounding data, to generate a contextualized, relevant, and accurate response.

### Understand how to ground your language model
Language models excel in generating engaging text, and are ideal as the base for agents. Agents provide users with an intuitive chat-based application to receive assistance in their work. When designing an agent for a specific use case, you want to ensure your language model is grounded and uses factual information that is relevant to what the user needs.

Though language models are trained on a vast amount of data, they may not have access to the knowledge you want to make available to your users. To ensure that an agent is grounded on specific data to provide accurate and domain-specific responses, you can use Retrieval Augmented Generation (RAG).

#### Understanding RAG
RAG is a technique that you can use to ground a language model. In other words, it's a process for retrieving information that is relevant to the user's initial prompt. In general terms, the RAG pattern incorporates the following steps:

1. **Retrieve grounding data** based on the initial user-entered prompt
2. **Augment the prompt** with grounding data
3. **Use a language model** to generate a grounded response
By retrieving context from a specified data source, you ensure that the language model uses relevant information when responding, instead of relying on its training data.

Using RAG is a powerful and easy-to-use technique for many cases in which you want to ground your language model and improve the factual accuracy of your generative AI app's responses.

### Make your data searchable
When you want to create an agent that uses your own data to generate accurate answers, you need to be able to search your data efficiently. When you build an agent with the Azure AI Foundry, you can use the integration with Azure AI Search to retrieve the relevant context in your chat flow.

Azure AI Search is a retriever that you can include when building a language model application with prompt flow. Azure AI Search allows you to bring your own data, index your data, and query the index to retrieve any information you need.


#### Using a vector index
While a text-based index will improve search efficiency, you can usually achieve a better data retrieval solution by using a vector-based index that contains embeddings that represent the text tokens in your data source.

An embedding is a special format of data representation that a search engine can use to easily find the relevant information. More specifically, an embedding is a vector of floating-point numbers.

For example, imagine you have two documents with the following contents:

- *"The children played joyfully in the park."*
- *"Kids happily ran around the playground."*

These two documents contain texts that are semantically related, even though different words are used. By creating vector embeddings for the text in the documents, the relation between the words in the text can be mathematically calculated.

The distance between vectors can be calculated by measuring the cosine of the angle between two vectors, also known as the cosine similarity. In other words, the cosine similarity computes the semantic similarity between documents and a query.

By representing words and their meanings with vectors, you can extract relevant context from your data source even when your data is stored in different formats (text or image) and languages.


## Frequently Asked Questions (FAQs)

### 🤖 General AI and Azure AI

**Q: What is the difference between traditional AI and Generative AI?**  
**A:** Traditional AI typically performs specific tasks like classification or prediction based on existing data patterns. Generative AI creates new, original content (text, images, code) from natural language prompts, making it more versatile for creative and content generation tasks.

**Q: What is Azure AI Foundry and how does it help developers?**  
**A:** Azure AI Foundry is a unified platform for building, evaluating, deploying, and managing AI solutions on Azure. It provides integrated tools, model hosting, and services that allow developers to add AI features without building everything from scratch.

**Q: Which Azure AI service should I use for my specific use case?**  
**A:** It depends on your needs:
- 🗨️ **Azure OpenAI**: For chat, content generation, and RAG applications
- 👁️ **Azure AI Vision**: For image analysis and computer vision tasks
- 🎤 **Azure AI Speech**: For voice assistants and speech processing
- 📝 **Azure AI Language**: For text analytics and document processing
- 🔍 **Azure AI Search**: For enterprise search and RAG grounding

---

### 🎯 Model Selection and Deployment

**Q: How do I choose between different language models like GPT-4, Mistral Large, or reasoning models?**  
**A:** Consider these factors:
- 🧠 **Task complexity**: Use reasoning models (DeepSeek-R1, o1) for complex math, coding, science tasks
- 🖼️ **Modality needs**: Choose multi-modal models (GPT-4o, Phi3-vision) if you need image processing
- ⚡ **Performance requirements**: GPT-4 and Mistral Large for high-quality text generation
- 💰 **Cost considerations**: Smaller models may be sufficient for simpler tasks

**Q: What's the difference between serverless endpoints and managed deployment?**  
**A:** 
- **Serverless endpoints**: Automatically scale based on demand and you pay per use, making them ideal for variable workloads
- **Managed deployment**: Gives you more control over infrastructure but requires you to manage scaling and availability

**Q: When should I fine-tune a model vs. using RAG?**  
**A:** 
- 🎨 **Fine-tuning**: When you need the model to learn specific writing styles, formats, or domain-specific language patterns
- 📚 **RAG**: When you need to ground responses in current, factual data or when your knowledge base frequently updates

---

### ✨ Prompt Engineering and Optimization

**Q: What are the most effective prompt engineering techniques?**  
**A:** Key techniques include:
- 📋 **Template specification**: Provide clear format structures for responses
- 🎯 **Context addition**: Include relevant background information
- ❓ **Question refinement**: Ask the model to suggest better ways to phrase queries
- 👤 **Role assignment**: Give the model a specific role (e.g., "You're a helpful AI assistant for sports reporters")

**Q: How can I improve the accuracy of my AI responses?**  
**A:** Use these strategies:
- 🔗 **RAG implementation**: Ground responses in relevant data sources
- 🛠️ **Prompt engineering**: Provide clear context and examples
- 🎯 **Model fine-tuning**: Train on domain-specific examples
- ✅ **Response validation**: Implement confidence thresholds and fallback logic

---

### 📊 RAG and Data Integration

**Q: What is RAG and when should I use it?**  
**A:** RAG (Retrieval Augmented Generation) retrieves relevant information from your data sources to ground AI responses in factual context. Use RAG when you need responses based on specific knowledge domains or current information not in the model's training data.

**Q: How do vector embeddings improve search accuracy?**  
**A:** Vector embeddings represent text as mathematical vectors that capture semantic meaning. This allows the system to find semantically related content even when different words are used, improving search relevance compared to traditional text-based search.

**Q: What's the difference between text-based and vector-based search indexes?**  
**A:** 
- 📝 **Text-based indexes**: Match exact keywords and phrases
- 🧮 **Vector-based indexes**: Understand semantic meaning and context, finding related content even with different terminology

---

### 🔄 Development Lifecycle

**Q: What are the key stages in developing an LLM application?**  
**A:** The four main stages are:

1. 🎯 **Initialization**: Define objectives and design the solution
2. 🧪 **Experimentation**: Develop and test flows with sample data
3. 📈 **Evaluation and refinement**: Test with larger datasets and optimize
4. 🚀 **Production**: Deploy, monitor, and continuously improve

**Q: How should I approach testing my LLM application?**  
**A:** Follow this approach:
- 📊 Start with a small, diverse sample dataset
- 🔄 Test iteratively during experimentation
- 📈 Use larger datasets for evaluation
- ⚡ Always test smaller datasets first before scaling up
- 📱 Monitor performance in production with user feedback

---

### ⚖️ Responsible AI

**Q: What are the main principles of Responsible AI?**  
**A:** The six core principles are:

| Principle | Focus | Description |
|-----------|-------|-------------|
| ⚖️ **Fairness** | Avoid bias | Ensure equitable outcomes for all users |
| 🛡️ **Reliability & Safety** | Consistent behavior | Ensure bounded, predictable responses |
| 🔒 **Privacy & Security** | Protect data | Safeguard user information and content |
| 🌍 **Inclusiveness** | Universal access | Design for all users and use cases |
| 🔍 **Transparency** | Clear communication | Explain system purpose and limitations |
| 👥 **Accountability** | Human oversight | Maintain governance and responsibility |

**Q: How can I ensure my AI application is fair and unbiased?**  
**A:** Implement these practices:
- 📊 Curate representative, diverse training data
- 🧪 Test across different demographic segments
- 🔧 Use bias mitigation techniques
- 📋 Regular auditing and monitoring
- 👥 Diverse user testing and feedback collection

