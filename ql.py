import random
import numpy as np
#import matplotlib.pyplot as plt #descomentar esta linea para ver la grafica

# Environment size
width = 5
height = 16

# Actions
num_actions = 4

#traduce entre la accion y el codigo correspondiente (0,1,2,3)
actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }
# coordenadas de las acciones
actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# traduccion inversa de las acciones por posicion
actions_position = { 0: "UP",
                     1: "RIGHT",
                     2: "DOWN",
                     3: "LEFT"
                    }

# Discount factor
discount = 0.8

# matriz q de estados, acciones
Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension

# devuelve el indice de un estado
def getState(y, x):
    return y * width + x

# devuelve las coordenadas de un indice
def getStateCoord(state):
    return int(state / width), int(state % width)

# numero de acciones posibles desde un estado
def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

#dado un estado elige una accion al azar
def getRndAction(state):
    return random.choice(getActions(state))

# coge cualquier estado al azar
def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000
# recompensa chachi
Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))

# funcion qlearning
# max(Q[s2]) es una lista/vector -> el maximo de la fila
def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return
state = 10
print np.argmax(Q[state])

# Episodes NORMAL
# numero promedio de acciones (198-256)
def normal():
    numPromAccion = 0
    for i in xrange(100):
        # partir de un estado aleatorio
        state = getRndState()
        # mientras no sea estado final...
        while state != final_state:
            # elegimos accion aleatoria
            action = getRndAction(state)
            # dada la accion, se mueve a un nuevo estado
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            # actualizamos tabla q
            qlearning(state, actions_list[action], new_state)
            # cambiamos el estado actual
            state = new_state
            #aumentamos uno por cada accion
            numPromAccion = numPromAccion + 1

    print Q
    #imprimimos el numero promedio
    print numPromAccion/100

# Episodes GREEDY
def greedy():
    # numero promedio de acciones
    numPromAccion = 0
    for i in xrange(100):
        # partir de un estado aleatorio
        state = getRndState()
        # mientras no sea estado final...
        while state != final_state:
            # elegimos accion segun info
            # pillamos maximo de lista de acciones y obtenemos el indice de la tabla
            if max(Q[state]) != 0 or (min(Q[state]) != 0 and max(Q[state] > min(Q[state]))):
            #if max(Q[state]) > 0:
                # devuelve posicion 0,1,2,3
                action = actions_position[np.argmax(Q[state])]
            else: # continuamos por una al azar porque no tenemos informacion
                action = getRndAction(state)
            # dada la accion, se mueve a un nuevo estado a partir de la nueva accion
            # getStateCoord[n] -> coordenadas del estado actual(y,x) y elegimos una de ellas (n)
            # actions_vectors -> la nueva accion
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            # actualizamos tabla q
            qlearning(state, actions_list[action], new_state)
            # cambiamos el estado actual
            state = new_state
            #aumentamos uno por cada accion
            numPromAccion = numPromAccion + 1

    print Q
    #imprimimos el numero promedio
    print numPromAccion/100

# Episodes E-GREEDY
def e_greedy(prob):
    # numero promedio de acciones
    numPromAccion = 0
    for i in xrange(100):
        # partir de un estado aleatorio
        state = getRndState()
        # mientras no sea estado final...
        while state != final_state:
            if max(Q[state]) != 0 or (min(Q[state]) != 0 and max(Q[state] > min(Q[state]))):
            # if max(Q[state]) > 0:
                # devuelve posicion 0,1,2,3
                if(random.randint(0,1) >= prob):
                    action = getRndAction(state)
                else:
                    action = actions_position[np.argmax(Q[state])]
            else: # continuamos por una al azar porque no tenemos informacion
                action = getRndAction(state)
            # dada la accion, se mueve a un nuevo estado
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            # actualizamos tabla q
            qlearning(state, actions_list[action], new_state)
            # cambiamos el estado actual
            state = new_state
            #aumentamos uno por cada accion
            numPromAccion = numPromAccion + 1

    print Q
    #imprimimos el numero promedio
    print numPromAccion/100



#Comentar y descomentar el metodo que se quiera usar
#normal()
#greedy()
e_greedy(0.8)

# Q matrix plot: imprime la matriz

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()
