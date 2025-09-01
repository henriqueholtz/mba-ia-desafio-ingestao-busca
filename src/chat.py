from search import search_prompt

def main():

    print("Faça sua pergunta:")
    user_input = input()
    print("Você perguntou:", user_input)
    
    chain = search_prompt(user_input)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    # response = chain.run(user_input)
    # print("Resposta:", response)

if __name__ == "__main__":
    main()