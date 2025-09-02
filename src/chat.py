from search import search_prompt

def main():

    print("Faça sua pergunta:")
    user_input = input()

    answer_gemini = search_prompt(user_input)

    print("-"*75)
    print("PERGUNTA: " + user_input)
    print("RESPOSTA: " + answer_gemini.content)

if __name__ == "__main__":
    main()