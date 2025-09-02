from search import search_prompt

def main():
    user_input = input("Faça sua pergunta: ")

    answer_gemini = search_prompt(user_input)

    print("-"*75, "\n")
    print("PERGUNTA: " + user_input)
    print("RESPOSTA: " + answer_gemini.content)

if __name__ == "__main__":
    main()