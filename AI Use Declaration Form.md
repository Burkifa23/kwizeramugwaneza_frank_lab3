## **AI Use Declaration Form** 

**Course:** Introduction to Artificial Intelligence

**Lab Title:** Lab 3: Neural Networks: Feedforward Networks and the Training Process

**Student Name:** Kwizera Mugwaneza Frank

**Student ID:** 36432028 

**GitHub Repository Link:** [https://github.com/Burkifa23/kwizeramugwaneza_frank_lab3](https://github.com/Burkifa23/kwizeramugwaneza_frank_lab3) 

**Date Submitted:** 28/07/2026

## **1. AI Use Summary** 

|**1. AI Use Summary**||
|---|---|
|**Question**|**Student Response**|
|Did you use any AI tool for this lab?| |
|Estimated percentage of the work influenced by AI| |
|Did you attach evidence of AI use where applicable?| |



## **2. Details of AI Use** 

Complete the table below for each AI tool used. Add more rows where necessary. 

| **Name of AI Tool Used** | **Purpose for Using the Tool** | **Prompt or Instruction Given to the Tool** | **Part(s) of the Work Influenced by the Tool** | **How I Verified, Edited, or Improved the AI Output** |
| :--- | :--- | :--- | :--- | :--- |
| Gemini |Initial Prompt |You are a Machine Learning and Deep Learning Tutor Expert You must guide me to achieve each task To correct me where I am wrong. YOU MUST NEVER GIVE ME THE CODE. EVEN IF I ASK FOR IT! when I get it wrong, you must give me hints and also tell me where to search for information to correct myself | This Chat was use to review all Coding Parts of the assignment | I had the guides for all questions |
| Gemini| Review Part 1.1 | Guide me through 1.1 | Part 1.1 | I understood what I need to do in 1.1|
| Gemini| Correct Code| Is this correct?  # Generate the toy data x = np.linspace(-3, 3, 100) gausian_small_noise = np.random.normal(loc=0.0, scale=0.1, size=x.shape) y = np.sin(x) + gausian_small_noise  #   Plot it. plt.plot(x, y) plt.title("Toy Data") plt.show()  # Initialise w and b to small random values. w = np.random.randn() b = np.random.randn()  # learning rate lr = 0.05 loss_record = []  # TODO: Implement ONE gradient-descent step by hand, then loop it for ~200 steps: for step in range(200): y_hat = w * x + b loss = np.mean((y_hat - y) ** 2) dw = np.mean(2 * (y_hat - y) * x) db = np.mean(2 * (y_hat - y)) w -= lr * dw b -= lr * db loss_record.append(loss)   # Plot (a) the loss curve over the 200 steps, and plt.plot(loss_record) plt.title("Loss Curve") plt.show() # (b) the data with the fitted line on top.  plt.plot(x, y) plt.plot(x, y_hat) plt.title("Fitted Line") plt.show() | Part 1.1 | I learned that When visualizing noisy data points, it is standard practice to use a scatter plot. Try changing plt.plot(x, y) to plt.scatter(x, y, label="Data") for your data, and keep plt.plot(x, y_hat, color='red', label="Fitted Line") for your neuron's prediction. And this I added this correction to my code|
| | | | | |
| | | | | |

## **3. Attachment of AI Output Evidence** 

Where required, attach evidence of AI use. Tick all that apply. Add more rows where necessary. 

|**Evidence Type**|**Attached?**<br>**(Yes, No,**<br>**N/A)**|**File Name**|
|---|---|---|
| | | |
| | | |


## **4. Student Declaration** 

I declare that: 

|**Response Declaration Statement**| **(Yes/No)** |
|---|---|
|The submitted work is my own work. | |
|Any use of AI tools has been clearly declared in this form. | |
|The prompts, instructions, and AI-generated outputs have been disclosed where applicable. | |
|I reviewed, tested, edited, and improved any AI-generated content before submission. |  |
|My AI usage does not exceed 25% of the entire work.|  |
|I understand that undeclared or excessive AI use may be treated as academic misconduct. |  |

**Student Signature:** _frank_

**Date:** 28/07/2026
