from typing import Optional
import sqlite3
import os
import json
import base64
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.db")
DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0) #ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# vision_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)
# vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

@tool
def search_products(query: str, max_price: Optional[float]=None, is_organic: Optional[bool]=None) -> str :
    """
    Search the product database by keyword (match against name, description and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price, description, is_organic.
    
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? or category LIKE ?)"
        like =f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += "AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += "AND is_organic = ?"
        params.append(1 if is_organic else 0)

    
    cursor.execute(sql,params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "description": row[4],
            "is_organic": row[5]
        }
        for row in rows
    ]
    return json.dumps(products)

@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a given product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)

# @tool
def checkout(product_id: int) -> str:
    """
    place an order for the given product ID. Saves the order to the database and 
    returns a confirmation message with the order ID, product name and price.

    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute("INSERT INTO Orders(product_id, product_name, price) VALUES(?, ?, ?)",
     (product_id, name, price)
    )

    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return(

        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f} "
        f"Your order will arrive in 3-5 business days. Thank you for ordering with us!"
    )

@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)
    mime = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"
    #print(image_data[:200])
    message = HumanMessage(content = [
        
        {
            "type":"text",
            "text":(
                """
                Look at this product image and extract its key attributes.
                Return ONLY a JSON object with these fields:
                    - product_type: what kind of product it is(eg., honey, olive oil, almonds ) \n
                    - search_query: a short keyword to search for it(eg., honey, olive oil ) \n
                    - is_organic: true if the label says organic, false if not, null if unclear.\n
                    - description: one sentense describing the product.
                """
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"}
        },
    ]

    )

    response = vision_llm.invoke([message])
    return response.content



# ----------------------------------
#      Agent
#-----------------------------------
SYSTEM_PROMPT = """
You are a helpful shopping assistant. Follow these rules strictly.
IMAGE SEARCH - when the user provides an image path:
1. Call describe_product_image  with the path to identify the product.
2. Use the returned search_query and is_organic to call search_products.
3. Continue with the BROWSING flow from step 2 onwards.
BROWSING - when the user describes what they want to buy:
1. Call search_products to find matching items(apply any price/organic filters given).
2. For each candidate, call get_rating to retrieve its average rating.
3. Filter by the user's minimum rating if specified.
4. Present qualifying products as a numbered list. For each item use this exact format:
    (plain text, no backticks, no code blocks, no bold, no italic):
    #<number>, <name> (ID:<product_id>) - <price> * <rating> - <organic or in-organic>
    Add a blank line between each product entry for readability.
    Always include <ID:X> so you can reference it later.
5. If only one product qualifies, still show it in the list and ask:
    would you like to order it? Just say yes or give me the number.

6. DO NOT call checkout at this stage.

ORDERING - when user confirms they want to buy (eg., 'yes', 'sure', 'go ahead' ),
  'order number 2', 'the first one', 'get me #3':
  1. Look at your previous message to find the (ID:X) for the chosen product.
    (if only one is listed and user says 'yes' use the product's ID).
  2. Call checkout with that product_id (the number from (ID:X)).
  3. Confirm the order to the user in plain text.
  NEVER place an order unless the user explicitly confirms.
  NEVER guess a product_id - always take it from the (ID:X)   in your own previous messages.



"""
agent = create_agent(
    model = llm,
    tools = [search_products, get_rating, checkout, describe_product_image],
    system_prompt=SYSTEM_PROMPT 
)



if __name__ == "__main__":
    # result = checkout(2)
    # print(result)
    # image_path = os.path.join("resources", "Honey.png")
    # #image_path = os.path.join(os.path.dirname(__file__), os.path.join("resources","Honey.png"))
    # #print(image_path)
    # response = describe_product_image(image_path)
    # print("Image description response:", response)


    result = agent.invoke(
        {
            "messages": [
                {
                    "role":"user",
                    "content":(
                        #"I want to buy an organic honey with 4.5+ rating and less than $20 price."
                        "I want to buy a tiger with 4.5+ rating and less than $20 price."
                    )
                }
            ]
        }
    )

    print(result["messages"][-1].content)

