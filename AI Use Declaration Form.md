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
|Gemini | Correction of 1.2 | Now This is what I did based on the pseudocode in What is wrong here? What can be done much better?  # Initialise parameters np.random.seed(42) # Reshape X = x.reshape(-1, 1) Y = y.reshape(-1, 1) #   W1: shape (1, 8),  b1: shape (8,) W1 = np.random.randn(1, 8) * 0.5 b1 = np.zeros(8) #   W2: shape (8, 1),  b2: shape (1,) #   Small random values (e.g. np.random.randn(...) * 0.5) W2 = np.random.randn(8, 1) * 0.5 b2 = np.zeros(1)  losses_tanh = []  lr = 0.05 for i in range(3000): # Forward pass (keep the intermediate values — you need them for backprop): #   z1 = x @ W1 + b1        # pre-activation, shape (100, 8) z1 = X @ W1 + b1 #   h  = np.tanh(z1)        # activation h = np.tanh(z1) #   y_hat = h @ W2 + b2     # output, shape (100, 1) y_hat = h @ W2 + b2 #   loss  = np.mean((y_hat - y) ** 2) loss = np.mean((y_hat - Y) ** 2)  # TODO: Backward pass (the chain rule, layer by layer): #   d_yhat = 2 * (y_hat - y) / len(y) d_yhat = 2 * (y_hat - Y) / len(Y) #   dW2 = h.T @ d_yhat dW2 = h.T @ d_yhat # db2 = d_yhat.sum(axis=0) db2 = d_yhat.sum(axis=0) #   dh  = d_yhat @ W2.T dh = d_yhat @ W2.T #   dz1 = dh * (1 - np.tanh(z1) ** 2)     # derivative of tanh dz1 = dh * (1 - np.tanh(z1) ** 2) #   dW1 = x.T @ dz1 ; dW1 = X.T @ dz1 # db1 = dz1.sum(axis=0) db1 = dz1.sum(axis=0)  # TODO: Update all parameters with learning rate lr, loop for ~3000 steps, W1 -= lr * dW1 b1 -= lr * db1 W2 -= lr * dW2 b2 -= lr * db2 #   record the loss losses_tanh.append(loss)  # Plot the loss curve AND the final fit over the data. plt.plot(losses_tanh) plt.title("Loss Curve") plt.xlabel("Step") plt.ylabel("MSE") plt.show() # Final Fit plt.scatter(X, Y) plt.plot(X, y_hat) plt.title("Final Fit") plt.show()   # tanh REMOVED (h = z1). What happens to the fit?   # Initialise parameters again W1 = np.random.randn(1, 8) * 0.5 b1 = np.zeros(8) W2 = np.random.randn(8, 1) * 0.5 b2 = np.zeros(1)  losses_no_tanh = []  lr = 0.05 for i in range(3000): # Forward pass z1 = X @ W1 + b1 # TANH REMOVED h = z1 y_hat = h @ W2 + b2 loss = np.mean((y_hat - Y) ** 2) losses_no_tanh.append(loss)  # Backward pass d_yhat = 2 * (y_hat - Y) / len(Y) dW2 = h.T @ d_yhat db2 = d_yhat.sum(axis=0) dh = d_yhat @ W2.T dz1 = dh dW1 = X.T @ dz1 db1 = dz1.sum(axis=0)  # Update all parameters W1 -= lr * dW1 b1 -= lr * db1 W2 -= lr * dW2 b2 -= lr * db2  # Plot the loss curve AND the final fit over the data. plt.plot(losses_no_tanh) plt.title("Loss Curve") plt.xlabel("Step") plt.ylabel("MSE") plt.show()  # Final Fit plt.scatter(X, Y) plt.plot(X, y_hat) plt.title("Final Fit Without TANH") plt.show()|1.2 | I had to make a function and flatten my visualizations |
| Gemini | 1.3 code correction|Is there anything wrong with Part 1.3 # Reshaped X = x.reshape(-1, 1) Y = y.reshape(-1, 1)   def train_toy(lr): np.random.seed(42)  W1 = np.random.randn(1, 8) * 0.5 b1 = np.zeros(8) W2 = np.random.randn(8, 1) * 0.5 b2 = np.zeros(1)  losses = []  for i in range(3000): # Forward pass (with tanh) z1 = X @ W1 + b1 h = np.tanh(z1) y_hat = h @ W2 + b2  loss = np.mean((y_hat - Y) ** 2) losses.append(loss)  # Backward pass d_yhat = 2 * (y_hat - Y) / len(Y) dW2 = h.T @ d_yhat db2 = d_yhat.sum(axis=0) dh = d_yhat @ W2.T dz1 = dh * (1 - np.tanh(z1) ** 2) dW1 = X.T @ dz1 db1 = dz1.sum(axis=0)  # Parameter update W1 -= lr * dW1 b1 -= lr * db1 W2 -= lr * dW2 b2 -= lr * db2  return losses   # Train lr=0.001 losses_slow = train_toy(lr=0.001) # Train lr=0.05 losses_good = train_toy(lr=0.05) # Train lr=1.0 losses_fast = train_toy(lr=1.0)  # Plot the results plt.figure(figsize=(10, 6)) plt.plot(losses_slow, label="lr = 0.001 (Too Cold)") plt.title("The Effect of Learning Rate on Training Loss") plt.xlabel("Step") plt.ylabel("MSE Loss (Log Scale)") plt.show()  plt.plot(losses_good, label="lr = 0.05 (Just Right)") plt.title("The Effect of Learning Rate on Training Loss") plt.xlabel("Step") plt.ylabel("MSE Loss (Log Scale)") plt.show()  plt.plot(losses_fast, label="lr = 1.0 (Too Hot)") plt.title("The Effect of Learning Rate on Training Loss") plt.xlabel("Step") plt.ylabel("MSE Loss (Log Scale)") plt.show() | 1.3|I had to change how I Graphed the different Learning rates in a one graph. |
|Gemini |Correction for 2.1 |This is the code for 2.1 Is there anywhere I am wrong? # import sklearn from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler, LabelEncoder  # Load the obesity dataset (same OBESITY_URL as Lab 2). OBESITY_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv" df = pd.read_csv(OBESITY_URL)  # Encode categoricals exactly as you did in Lab 2 label_encoder = LabelEncoder() df["NObeyesdad"] = label_encoder.fit_transform(df["NObeyesdad"]) #   (binary yes/no -> 0/1; Gender/MTRANS/CAEC/CALC -> your justified choice). binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"] df[binary_cols] = df[binary_cols].replace({"yes": 1, "no": 0}) df["Gender"] = df["Gender"].replace({"Female": 0, "Male": 1})  # Ordinal encoding freq_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3} df["CAEC"] = df["CAEC"].map(freq_mapping) df["CALC"] = df["CALC"].map(freq_mapping)  #   Encode the target NObeyesdad into integers 0..6 (LabelEncoder) — CrossEntropyLoss #   expects integer class indices. df = pd.get_dummies(df, columns=["MTRANS"], drop_first=True) X = df.drop(columns=["NObeyesdad"]) y = df["NObeyesdad"]   # Stratified train / validation / test split (60/20/20, stratify=y, random_state=RANDOM_STATE). # First split: 60% train, 40% temp X_train, X_temp, y_train, y_temp = train_test_split( X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE )  # Second split: split the 40% temp into 20% validation and 20% test X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE )   # Fit a StandardScaler on the TRAINING set only; transform all three splits. #   (Neural networks are far more sensitive to feature scale than random forests remember Section 1, where all gradients depended on x!) scaler = StandardScaler() # Fitted on ONLY training data X_train_scaled = scaler.fit_transform(X_train) X_val_scaled = scaler.transform(X_val) X_test_scaled = scaler.transform(X_test)   # Convert to tensors and wrap in DataLoaders: #   X -> torch.float32, y -> torch.long X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32) y_train_t = torch.tensor(y_train.values, dtype=torch.long)  X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32) y_val_t = torch.tensor(y_val.values, dtype=torch.long)  X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32) y_test_t = torch.tensor(y_test.values, dtype=torch.long)  # Wrap in TensorDatasets train_dataset = TensorDataset(X_train_t, y_train_t) val_dataset = TensorDataset(X_val_t, y_val_t) test_dataset = TensorDataset(X_test_t, y_test_t) #   train_loader (batch_size=32, shuffle=True), val_loader, test_loader # Create DataLoaders train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}") print(f"Number of input features: {X_train_t.shape[1]}") |2.1 | The chat affirmed I was correct so no need to changeand gave me guide for 2.2 |
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
