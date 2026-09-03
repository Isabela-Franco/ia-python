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
    "Eu gosto de gatos",
    "Eu gosto de cachorros",
    "Eu gosoto de coelhos",
    "Eu gosto de pizza",
    "Eu gosto de Minecraft"
]

embeddings = modelo.encode(frases)
memoria = {}
historico = []
embeddings_assuntos = modelo.encode(assuntos)

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

    if len(mensagens) >= 2:
        return mensagens[-2]
    
    return None

def descobrir_assunto(texto):
    embeddings = modelo.encode([texto])

    similaridade = cosine_similarity(embeddings, embeddings_assuntos)[0]
    indice = similaridade.argmax()

    return assuntos[indice], similaridade[indice]

#ARQUIVO JSON
def salvar_memoria():
    with open("memoria.json", "w", encoding="utf-8") as arquivo:
        json.dump(memoria, arquivo, ensure_ascii=False, indent=4)



#LOOP PRINCIPAL
while True:

    texto = input("Você: ")

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

        assunto, confianca = descobrir_assunto(texto)

        print("Assunto: ", assunto)
        print("Confiança: ", round(confianca, 2))

        if anterior:
            if "gato" in anterior.lower():
                print("IA: Eu também gosto de gatos!")
            elif "cachorro" in anterior.lower():
                print("IA: Cachorros são ótimos companheiros!")
            else:
                print(f"IA: Entendi! Você estava falando sobre isso {anterior}.")

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
    

