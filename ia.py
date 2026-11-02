from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random


modelo = SentenceTransformer('all-MiniLM-L6-v2')

#LISTAS
frases = [
    "Eu tenho um cachorro",
    "Eu gosto de gatos",
    "Até amanhã",
    "Obrigado pela ajuda",
    "Qual seu nome?",
    "Como você se chama?",
    "Quem é você?",
    "Me diga seu nome",
    "Como você está?",
    "Como você se sente?",
    "Está tudo bem?",
    "Tudo bem com você?",
    "Até logo",
    "Oi",
    "Olá",
    "Bom dia",
    "Boa tarde",
    "Boa noite",
    "Tudo bem?",
]

Categorias = [
    "animal",
    "animal",
    "despedida",
    "agradecimento", 
    "nome",
    "nome",
    "nome",
    "nome",
    "estado",
    "estado",
    "estado",
    "estado",
    "despedida",
    "saudaçâo",
    "saudaçâo",
    "saudaçâo",
    "saudaçâo",
    "saudaçâo",
    "saudaçâo",
]

respostas = {
    "animal": [
        "Que legal!",
        "Eu também gosto de animais!",
        "Animais são incríveis!"
    ],

    "agradecimento": [
        "Por nada!",
        "Disponha!",
        "Não foi nada!"
    ],

    "estado": [
        "Estou bem! E você?",
        "Tudo certo por aqui",
        "Estou ótimo! E você?"
    ],

    "despedida": [
        "Até mais!",
        "Tchau!",
        "Nos vemos em breve!"
    ],

    "saudaçâo": [
        "Oi! ",
        "Olá! Tudo bem?",
        "Oi! Que bom falar com você!",
        "Olá! Como você está?"
    ],

    
}

palavra_contexto = {
    "e você",
    "e vc",
    "por quê",
    "por que",
    "como assim",
}

assuntos = [
    ("Eu gosto de gatos", "gatos"),
    ("Eu tenho um gato", "gatos"),
    ("Eu adoro gatos", "gatos"),
    ("Meu animal favorito é o gato", "gatos"),

    ("Eu gosto de cachorros", "cachorros"),
    ("Eu tenho um cachorro", "cachorros"),
    ("Eu adoro cachorros", "cachorros"),
    ("Meu animal favorito é o cachorro", "cachorros"),

    ("Eu gosto de coelhos", "coelhos"),
    ("Eu tenho um coelho", "coelhos"),

    ("Eu gosto de pizza", "pizza"),
    ("Minha comida favorita é pizza", "pizza"),

    ("Eu gosto de Minecraft", "Minecraft"),
    ("Eu jogo Minecraft", "Minecraft"),
]

respostas_contexto = {
    "gatos": [
        "Eu também gosto de gatos!",
        "Gatos são muito legais!",
        "Eu acho gatos fascinantes."
    ],

    "cachorros": [
        "Cachorros são ótimos companheiros!",
        "Eu também acho cachorros muito legais!",
        "Cachorros têm uma energia incrível."
    ],

    "coelhos": [
        "Coelhos são muito fofos!",
        "Eu também acho coelhos adoráveis!"
    ],

    "pizza": [
        "Pizza é uma ótima escolha!",
        "Eu também gosto da ideia de uma boa pizza!"
    ],

    "Minecraft": [
        "Minecraft é muito divertido!",
        "Dá para fazer muita coisa no Minecraft!",
        "Minecraft tem possibilidades praticamente infinitas."
    ]
}

frases_assuntos = [item[0] for item in assuntos]
embeddings_assuntos = modelo.encode(frases_assuntos)
embeddings = modelo.encode(frases)
memoria = {}
historico = []

#FUNÇÂO CLASSIFICSAR TEXTO
def classificar_texto(texto):
    embedding_novo = modelo.encode([texto])
    similaridade = cosine_similarity(embedding_novo, embeddings)[0]

    indice = similaridade.argmax()
    categoria = Categorias[indice]
    confianca = similaridade[indice]

    return categoria, confianca

def historico_resposta(resposta):
    historico.append({
        "IA": resposta
    })

def ultima_mensagem_usuario():
    mensagens = []

    for item in historico:
        if "usuario" in item:
            mensagens.append(item["usuario"])
            
    print("Mensagens: ", mensagens)

    if len(mensagens) >= 2:
        print("Anterior", mensagens[-2])
        return mensagens[-2]
    
    return None

def descobrir_assunto(texto):
    embeddings = modelo.encode([texto])

    similaridade = cosine_similarity(embeddings, embeddings_assuntos)[0]
    indice = similaridade.argmax()

    indice = similaridade.argmax()
    categoria = assuntos[indice][1]

    return categoria, similaridade[indice]


def executar_acao(acao):
    if acao == "mostrar_memoria":
        if memoria:
            return f"Estas são as informações que eu lembro: {memoria}"
        else:
            return "Eu não tenho informações salvas."

    elif acao == "apagar_memoria":
        if memoria:
            chave = input("Qual informação deseja apagar? ").lower().strip()

            if chave in memoria:
                del memoria[chave]
                salvar_memoria()
                return f"Informação '{chave}' foi apagada!"
            else:
                return "Não encontrei essa informação."
        else:
            return "Eu não tenho informações salvas."

    return "Não conheço essa ação."


acoes = {
    "mostrar_memoria": executar_acao,
    "apagar_memoria": executar_acao
}

def chamar_acao(nome_acao):
    funcao = acoes.get(nome_acao)

    if funcao is None:
        return "Não conheço essa ação."

    return funcao(nome_acao)

#ARQUIVO JSON
def salvar_memoria():
    with open("memoria.json", "w", encoding="utf-8") as arquivo:
        json.dump(memoria, arquivo, ensure_ascii=False, indent=4)



#LOOP PRINCIPAL
while True:

    texto = input("Você: ")

    if "o que você lembra de mim?" in texto.lower():
        resposta = chamar_acao("mostrar_memoria")

        print(f"IA: {resposta}")
        historico_resposta(resposta)
        continue

    if "apagar informação da memória" in texto.lower():
        resposta = chamar_acao("apagar_memoria")
        print(f"IA: {resposta}")
        historico_resposta(resposta)
        continue

    historico.append({
        "usuario": texto
    })
    
    categoria, confianca = classificar_texto(texto)

    if "meu nome é" in texto.lower():
        posicao = texto.lower().index("meu nome é") + len("meu nome é")

        nome = texto[posicao:].strip()

        memoria["nome"] = nome

        salvar_memoria()

        resposta = f"Prazer em te conhecer, {nome}!"
        print(f"IA: {resposta}")
        historico_resposta(resposta)
        continue

    if "qual é o meu nome?" in texto.lower() or "qual o meu nome?" in texto.lower():
        if "nome" in memoria:
            resposta = f"Seu nome é {memoria['nome']}."
            print(f"IA: {resposta}")
            historico_resposta(resposta)
        else:
            print("IA: Eu não sei seu nome ainda. Me diga seu nome primeiro.")
        continue

    if "minha comida favorita é" in texto.lower():
        posicao = texto.lower().index("minha comida favorita é") + len("minha comida favorita é")

        comida = texto[posicao:].strip()

        memoria["comida"] = comida

        salvar_memoria()

        resposta = f"Que legal! Eu também gosto de {comida}."
        print(f"IA: {resposta}")
        historico_resposta(resposta)
        continue

    if "qual é a minha comida favorita?" in texto.lower() or "qual a minha comida favorita?" in texto.lower():
        if "comida" in memoria:
            resposta = f"Sua comida favorita é {memoria['comida']}."
            print(f"IA: {resposta}")
            historico_resposta(resposta)
        else:
            print("IA: Eu não sei sua comida favorita ainda. Me diga sua comida favorita primeiro.")
        continue

    if "meu jogo favorito é" in texto.lower():
        posicao = texto.lower().index("meu jogo favorito é") + len("meu jogo favorito é")

        jogo = texto[posicao:].strip()

        memoria["jogo"] = jogo

        salvar_memoria()

        resposta = f"Que legal! Eu também gosto de {jogo}."
        print(f"IA: {resposta}")
        historico_resposta(resposta)
        continue

    if "qual é o meu jogo favorito?" in texto.lower() or "qual o meu jogo favorito?" in texto.lower():
        if "jogo" in memoria:
            resposta = f"Seu jogo favorito é {memoria['jogo']}."
            print(f"IA: {resposta}")
            historico_resposta(resposta)
        else:
            print("IA: Eu não sei seu jogo favorito ainda. Me diga seu jogo favorito primeiro.")
        continue

    if any(palavra in texto.lower() for palavra in palavra_contexto):
        anterior = ultima_mensagem_usuario()


        if anterior:
            assunto, confianca = descobrir_assunto(anterior)
            print("Assunto: ", assunto)
            print("Confiança: ", confianca)

            if confianca < 0.55:
                print("IA: Não consegui entender qual é o assunto.")

            else:
                if assunto in respostas_contexto:
                    resposta = random.choice(respostas_contexto[assunto])
                    print(f"IA: {resposta}")
                    historico_resposta(resposta)
                else:
                    print(f"IA: Entendi! Você estva falando sobre {assunto}.")  

                continue

    if confianca < 0.60:
        print("IA: Não entendi muito bem.")

    elif categoria in respostas:
        resposta = random.choice(respostas[categoria])
        print(f"IA: {resposta}")

        historico_resposta(resposta)

    
    elif categoria == "nome":
        print("IA: Eu sou uma IA.")

#SAIR DO PROGRAMA
    if texto.lower() == "sair":
        print("Até mais!")
        print(historico)
        break
    

