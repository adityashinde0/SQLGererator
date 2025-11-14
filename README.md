 # SQL Query Generator 🤖

A powerful natural language to SQL query generator powered by the SQLCoder-7B-2 model. This project allows you to generate SQL queries from plain English questions using state-of-the-art language models.

## 🌟 Features

- **Natural Language Processing**: Convert plain English questions into SQL queries
- **Advanced Model**: Uses defog/sqlcoder-7b-2, a specialized model for SQL generation
- **GPU Acceleration**: Supports both full precision (16GB+ VRAM) and 8-bit quantization (lower VRAM)
- **Query Formatting**: Automatically formats generated SQL queries for readability
- **Easy to Use**: Simple Jupyter notebook interface

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab account (for cloud execution) or local GPU setup

## 🚀 Installation

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge at the top of the notebook
2. Run the first cell to install dependencies
3. Follow the notebook cells sequentially

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/adityashinde0/SQLGenerator.git
cd SQLGenerator

# Install dependencies
pip install torch transformers bitsandbytes accelerate sqlparse

# Launch Jupyter
jupyter notebook SQLGenerator.ipynb
```

## 💻 Usage

### Basic Example

```python
# Initialize the model (run once)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'defog/sqlcoder-7b-2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
)

# Generate SQL query
question = "What was the highest quantity sold last month?"
generated_sql = generate_query(question)
print(generated_sql)
```

### Sample Questions

```python
# Revenue analysis
"What was our revenue by product in New York?"

# Sales performance
"Which salesperson sold the largest amount of products last month?"

# Inventory queries
"What products have quantity less than 10?"

# Customer insights
"Who are the top 5 customers by total purchase amount?"
```

## 📊 Database Schema

The generator works with the following database schema:

- **products**: product_id, name, price, quantity
- **customers**: customer_id, name, address
- **salespeople**: salesperson_id, name, region
- **sales**: sale_id, product_id, customer_id, salesperson_id, sale_date, quantity
- **product_suppliers**: supplier_id, product_id, supply_price

## 🔧 Configuration

### Memory Management

The notebook automatically detects available GPU memory:
- **>16GB VRAM**: Uses float16 precision for faster inference
- **<16GB VRAM**: Uses 8-bit quantization to reduce memory usage

### Model Parameters

You can adjust generation parameters in the `generate_query()` function:
```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=400,      # Maximum length of generated SQL
    num_beams=1,             # Beam search width
    do_sample=False,         # Deterministic generation
)
```

## 📝 Example Outputs

**Question**: "What was the highest quantity sold last month?"

**Generated SQL**:
```sql
SELECT MAX(quantity)
FROM sales
WHERE sale_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH);
```

**Question**: "Which salesperson sold the largest amount of products last month?"

**Generated SQL**:
```sql
SELECT sp.name,
       SUM(s.quantity) AS total_quantity
FROM sales s
JOIN salespeople sp ON s.salesperson_id = sp.salesperson_id
WHERE s.sale_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)
GROUP BY sp.salesperson_id,
         sp.name
ORDER BY total_quantity DESC
LIMIT 1;
```

## ⚠️ Limitations

- Requires GPU for optimal performance
- Model needs ~15GB VRAM in full precision or ~8GB with quantization
- Generated queries should be reviewed before production use
- Complex multi-step queries may require refinement

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- for the SQLCoder-7B-2 model
- Hugging Face for the transformers library
- The open-source community

## 📧 Contact

Aditya Shinde - [@adityashinde0](https://github.com/adityashinde0)

Project Link: [https://github.com/adityashinde0/SQLGenerator](https://github.com/adityashinde0/SQLGenerator)

## 🔗 Resources

- [SQLCoder Model Card](https://huggingface.co/defog/sqlcoder-7b-2)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [SQL Tutorial](https://www.w3schools.com/sql/)

---

⭐ If you find this project helpful, please consider giving it a star!
