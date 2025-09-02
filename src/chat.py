from search import search_prompt

def main():
    while True:
        if user_input := input("Faça a sua pergunta: "):
            answer_gemini = search_prompt(user_input)
            print("-"*75, "\n")
            print("PERGUNTA: " + user_input)
            print("RESPOSTA: " + answer_gemini.content)
            print("-"*75, "\n")
        else:
            print("Chat encerrado!")
            break


if __name__ == "__main__":
    main()