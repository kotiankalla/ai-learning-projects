from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()



# llm= ChatGoogleGenerativeAI(model="gemma-4-31b-it")
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)


def get_restaurant_name_and_items(cuisine):
    
    prompt_template_name = PromptTemplate(
        input_variables = ["cuisine"],
        template = "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. Give name ONLY. DONT include any other information "
    )

    #prompt_template_name.format(cuisine = "Italian")
    chain = prompt_template_name | llm

    # 2. Run it using .invoke() passing a dictionary
    response = chain.invoke({"cuisine": cuisine})
    # print(response.content)
    restaurant_name = response.content

    menu_items_prompt_template_name = PromptTemplate(
    input_variables = ["restaurant_name"],
    template = "Suggest some menu items for {restaurant_name}. Return it as a comma seperated string"
)
    menu_chain = menu_items_prompt_template_name | llm
    menu_response = menu_chain.invoke({"restaurant_name": restaurant_name})
    # print(response.content)
    menu_items = menu_response.content.strip()
    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

if __name__ == "__main__":
    output = get_restaurant_name_and_items("American")

    print("Restaurant Name:", output["restaurant_name"])
    print("Menu Items:", output["menu_items"])