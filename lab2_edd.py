from typing import List
import html
from queue import PriorityQueue
from typing import Tuple, List, Dict

import folium #Mapa
import webbrowser

import networkx as nx
import plotly.graph_objs as go #Grafo

import csv
notes = ''
with open('/Users/lymich/Documents/kes/TRABAJOS/4 SEMESTRE/EDD 2/LAB2/flights_final.csv', newline='') as f:
    data = csv.reader(f, delimiter=',')
    notes = list(data)

############### CLASE AEROPUERTO ###############

class Airport:
    def __init__(self, code, name, city, country, latitude, longitude):
        self.code = code
        self.name = name
        self.city = city
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self):
        return f"{self.name} ({self.code}). {self.city}, {self.country}.\n  Lat: {self.latitude}, Long: {self.longitude}"

    # Definir el método __lt__ para comparar aeropuertos por su código
    def __lt__(self, other):
        return self.code < other.code

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)  # Para que los objetos sean hashable y puedan ser usados en sets y diccionarios

class Flight:
    def __init__(self, source_airport_code, source_airport_name, source_airport_city, source_airport_country,
                 source_airport_latitude, source_airport_longitude, destination_airport_code,
                 destination_airport_name, destination_airport_city, destination_airport_country,
                 destination_airport_latitude, destination_airport_longitude):
        # Source airport attributes
        self.source_airport = Airport(source_airport_code, source_airport_name, source_airport_city,
                                      source_airport_country, float(source_airport_latitude), float(source_airport_longitude))
        # Destination airport attributes
        self.destination_airport = Airport(destination_airport_code, destination_airport_name, destination_airport_city,
                                           destination_airport_country, float(destination_airport_latitude), float(destination_airport_longitude))

    def __repr__(self):
        return (f"Flight from {self.source_airport_name} ({self.source_airport_code}) to {self.destination_airport_name} "
                f"({self.destination_airport_code})")


def get_flight_data(notes):
    flights_list = []

    for i in range(1, len(notes)):
        flight = Flight(
            source_airport_code = notes[i][0],
            source_airport_name = notes[i][1],
            source_airport_city = notes[i][2],
            source_airport_country = notes[i][3],
            source_airport_latitude = notes[i][4],
            source_airport_longitude = notes[i][5],
            destination_airport_code = notes[i][6],
            destination_airport_name = notes[i][7],
            destination_airport_city = notes[i][8],
            destination_airport_country = notes[i][9],
            destination_airport_latitude = notes[i][10],
            destination_airport_longitude = notes[i][11]
        )
        flights_list.append(flight)

    return flights_list

############### CLASE DEL GRAFO ###############

import math
class Graph:

    def __init__(self, directed: bool = False):
        #self.n = n
        self.directed = False
        self.L = {}  # Diccionario que mapea los aeropuertos y los vuelos
        self.cost_matrix = {}  # Diccionario para almacenar costos entre vuelos
        self.edges = []  # Lista para almacenar todas las aristas (vuelos)


    def add_vertex(self, aeropuerto: Airport) -> None:
        if aeropuerto not in self.L:
            self.L[aeropuerto] = []

    def add_edge(self, flight: Flight) -> bool:

        # Asegurarse de que ambos vuelos estén en el grafo
        u = flight.source_airport
        v = flight.destination_airport
        self.add_vertex(u)
        self.add_vertex(v)

        # Añadir vuelo adyacente y costo
        cost = self.calcular_costo(flight)

        self.L[u].append(v)
        self.cost_matrix[(u, v)] = cost  # Asignar costo a la matriz
        self.edges.append(flight)

        if not self.directed:
            self.L[v].append(u)
            self.cost_matrix[(v, u)] = cost  # Reflejar costo si es no dirigido

        return True
        pass

    def calcular_costo(self, flight1: Flight) -> float:

      long_source = flight1.source_airport.longitude
      lat_source = flight1.source_airport.latitude

      long_destination = flight1.destination_airport.longitude
      lat_destination = flight1.destination_airport.latitude

      R = 6372.795477598 #km

      dlong = math.radians(long_destination - long_source)
      dlat = math.radians(lat_destination - lat_source)

      a = (math.sin(dlat / 2))**2 + math.cos(math.radians(lat_source)) * math.cos(math.radians(lat_destination)) * (math.sin(dlong / 2))**2
      distancia = 2 * R * math.asin(math.sqrt(a))

      return distancia #se devuelve en km
      pass

    def get_cost_byAirports(self, airport1: Airport, airport2: Airport) -> float:
      return self.cost_matrix.get((airport1, airport2), 0)  # Retorna 0 si no hay conexión directa

    def get_cost_byFlight(self, flight: Flight) -> float:
      airport1 = flight.source_airport
      airport2 = flight.destination_airport
      return self.cost_matrix.get((airport1, airport2), 0)  # Retorna 0 si no hay conexión directa

    def DFS(self, start: Airport) -> None:
        visit = {airport: False for airport in self.L}  # Crear un diccionario en vez de una lista por indice para marcar los aeropuertos visitados
        self.__DFS_visit(start, visit)

    def __DFS_visit(self, u: Airport, visit: dict) -> None:
        visit[u] = True
        print(u.code, end=' ')  # Imprimir el código del aeropuerto, se puede cambiar si lo necesitamos
        for v in self.L[u]:
            if not visit[v]:
                self.__DFS_visit(v, visit)

    def BFS(self, u: Airport) -> None:
        queue = []
        visit = {airport: False for airport in self.L}  # Crear un diccionario para marcar los aeropuertos visitados
        visit[u] = True
        queue.append(u)
        while queue:
            u = queue.pop(0)
            print(u.code, end=' ')  # Imprimir el código del aeropuerto, se puede cambiar si lo necesitamos
            for v in self.L[u]:
                if not visit[v]:
                    visit[v] = True
                    queue.append(v)

    def degree(self, u: Airport) -> int:
        return len(self.L[u])
        pass

############### ARBOL DE EXPANSIÓN PARA CADA COMPONENTE ###############

    def number_of_components(self) -> Tuple[int, List[List[Airport]]]:
        visit = {u: False for u in self.L}  # Cambiamos la lista de visitados por un diccionario
        count = 0
        components = []  # Lista para almacenar los vertices de cada componente

        for u in self.L:
            if not visit[u]:
                count += 1
                component = []
                visit = self.__DFS_components(u, visit, component) #hay que guardar la lista, el metodo DFS devuelve una lista
                components.append(component)
        return count, components
        pass

    def __DFS_components(self, u: Airport, visit: dict, vertices: List[Airport]) -> dict:
        visit[u] = True
        vertices.append(u)
        for v in self.L[u]:
            if not visit[v]:
                visit = self.__DFS_components(v, visit, vertices)
        return visit
    
    def kruskal_cada_componente(self) -> List[Tuple[List[Airport], float]]:
        num_comp, components = self.number_of_components()
        results = []

        for component in components:
            Arbol_expansion, total_weight = self.Kruskal(component)
            results.append((Arbol_expansion, total_weight))  # Arbol expansion de la componente y su peso

        return results
        pass

    def grafo_conexo_texto(self) -> str:
        result_str = ""
        num_components, components = self.number_of_components()
        if num_components == 1:
            result_str += "El grafo es conexo. Tiene 1 componente."
        else:
            result_str += f"El grafo no es conexo. Número de componentes conexas: {num_components}\n\n"
            for i, component in enumerate(components, 1):
                result_str += (f"Componente {i}: {len(component)} vértices\n")

        return result_str
        pass

    def kruskal_costos_porArbol_texto(self) -> str:
        result_str = ""
        for i, (_, weight) in enumerate(self.kruskal_cada_componente(), 1):
            result_str += f"Peso del árbol de expansión mínima {i}: {weight}\n"

        return result_str
        pass

############### METODO KRUSKAL ###############

    def find(self, parent: Dict[Flight, Flight], flight: Flight) -> Flight:
        if parent[flight] == flight:
            return flight
        parent[flight] = self.find(parent, parent[flight])  # Compresión de camino
        return parent[flight]

    def union(self, parent: Dict[Flight, Flight], rank: Dict[Flight, int], flight1: Flight, flight2: Flight):
        root1 = self.find(parent, flight1)
        root2 = self.find(parent, flight2)

        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1

    def Kruskal(self, component: List[Airport]) -> List[Tuple[Airport, Airport, float]]:
        # Cola de prioridad para las aristas (peso, flight1, flight2)
        q = PriorityQueue()

        # Añadir todas las aristas a la cola de prioridad
        for (u, v), cost in self.cost_matrix.items():
            if u in component and v in component and u.code < v.code:  # Evitar duplicar aristas en grafos no dirigidos
                q.put((cost, u, v))  # Las aristas son (u, v) con el costo entre ellos

        # Estructura union-find
        parent = {airport: airport for airport in component}  # Cada vuelo es su propio padre
        rank = {airport: 0 for airport in component}  # Inicializar los rangos

        # Árbol de expansión mínima
        T = []
        total_weight = 0
        num_components, b = self.number_of_components()
        while len(T) < len(component) - 1 and not q.empty():
            # Tomar la arista de menor peso
            cost, u, v = q.get()

            # Verificar si forman un ciclo
            root_u = self.find(parent, u)
            root_v = self.find(parent, v)

            if root_u != root_v:
                T.append((u, v, cost))  # Añadir arista al árbol
                total_weight += cost  # Sumar el costo al peso total
                self.union(parent, rank, root_u, root_v)  # Unir conjuntos

        return T, total_weight

############### CAMINOS MINIMOS ###############
    
    def dijkstra(self, Vertice_inicial: Airport) -> Tuple[Dict[Airport, float], Dict[Airport, Airport]]:
        D = {airport: float('inf') for airport in self.L}
        pad = {airport: None for airport in self.L}
        visit = {airport: False for airport in self.L}
        D[Vertice_inicial] = 0

        while not all(visit.values()):
            v = min((airport for airport in self.L if not visit[airport]), key=lambda airport: D[airport])
            visit[v] = True

            for neighbor in self.L[v]:
                if not visit[neighbor] and D[v] + self.cost_matrix[(v, neighbor)] < D[neighbor]:
                    D[neighbor] = D[v] + self.cost_matrix[(v, neighbor)]
                    pad[neighbor] = v

        return D, pad

    def dijkstra_dosVertices(self, vertice_inicial: Airport, vertice_final: Airport) -> Tuple[Dict[Airport, float], Dict[Airport, Airport]]:
        D, pad = self.dijkstra(vertice_inicial)

        #Reconstruye el camino mínimo desde el vértice final al inicial
        path = []
        current = vertice_final
        while current is not None:
            path.insert(0, current)
            current = pad[current]
        
        if path[0] != vertice_inicial:
            return []

        return path
    
    def mostrar_camino_en_mapa(self, camino: List[Airport]):
        if not camino:
            print("No hay camino para mostrar.")
            return

        #Para crear el mapa
        mapa = folium.Map(location=[camino[0].latitude, camino[0].longitude], zoom_start=5)

        #Añadir los aeropuertos del camino
        for i, aeropuerto in enumerate(camino):
            folium.Marker([aeropuerto.latitude, aeropuerto.longitude],
                popup=folium.Popup((f"Código: {aeropuerto.code}<br>"
                   f"Nombre: {aeropuerto.name}<br>"
                   f"Ciudad: {aeropuerto.city}<br>"
                   f"País: {aeropuerto.country}<br>"
                   f"Latitud: {aeropuerto.latitude}<br>"
                   f"Longitud: {aeropuerto.longitude}"), max_width = 300),
                   icon=folium.Icon(color="orange")
            ).add_to(mapa)

            if i < len(camino) - 1:
                next_airport = camino[i + 1]
                folium.PolyLine([(aeropuerto.latitude, aeropuerto.longitude),
                                (next_airport.latitude, next_airport.longitude)],
                                color="black", weight=2.5).add_to(mapa)
                
        mapa.save("camino_minimo.html")
        webbrowser.open(f"file://{os.path.abspath("camino_minimo.html")}")
    
    def mostrar_mapa(self):
        mapa = folium.Map(zoom_start=5)

        for aeropuerto in self.L:
            folium.Marker([aeropuerto.latitude, aeropuerto.longitude],
                popup=folium.Popup(
                    (f"Código: {aeropuerto.code}<br>"
                    f"Nombre: {aeropuerto.name}<br>"
                    f"Ciudad: {aeropuerto.city}<br>"
                    f"País: {aeropuerto.country}<br>"
                    f"Latitud: {aeropuerto.latitude}<br>"
                    f"Longitud: {aeropuerto.longitude}"), max_width=300),
                    icon=folium.Icon(color="orange")
            ).add_to(mapa)

        mapa.save("mapa_general.html")
        webbrowser.open(f"file://{os.path.abspath("mapa_general.html")}")

#############

    def dijkstra_componente(self, Vertice_inicial: Airport, component: List[Airport]) -> Tuple[Dict[Airport, float], Dict[Airport, Airport]]:

        D = {Airport: float('inf') for Airport in component}
        pad = {Airport: None for Airport in component}
        visit = {Airport: False for Airport in component}
        D[Vertice_inicial] = 0

        while not all(visit.values()):
            v = min((Airport for Airport in component if not visit[Airport]), key=lambda Airport: D[Airport])
            visit[v] = True  # Marca el nodo como visitado

            for i in self.L[v]:
                if i in component and not visit[i] and D[v] + self.cost_matrix[(v, i)] < D[i]:
                    D[i] = D[v] + self.cost_matrix[(v, i)]  # Actualiza la distancia mínima
                    pad[i] = v  # Actualiza el predecesor de i

        return D, pad  # Retorna las distancias y los predecesores"""

    def caminos_desde(self, Vertice_inicial: Airport) -> str:
        _ , components = self.number_of_components()
        for component in components:
            for u in component:
                if u == Vertice_inicial:
                    D, pad = self.dijkstra_componente(u, component)
                    break
        
        resultado_str = f"{Vertice_inicial.name} ({Vertice_inicial.code})\n{Vertice_inicial.city}, {Vertice_inicial.country}\nLatitude: {Vertice_inicial.latitude}, Longitude: {Vertice_inicial.longitude}\n"
        List = []

        for airport, distance in D.items():
            if airport != Vertice_inicial:
                List.append((airport, distance, pad[airport]))

        # Ordenar la lista por las distancias de mayor a menor
        List = sorted(List, key=lambda x: x[1], reverse=True)
        resultado_str2 = ""
        for i in range(min(10, len(List))):
            resultado_str2 += f"- {i+1}): {List[i][0]}, Distancia: {List[i][1]}\n"
        return resultado_str, resultado_str2

#############

graph = Graph(directed=False)
flights = get_flight_data(notes)

for flight in flights:
    graph.add_edge(flight)

def obtener_por_codigo(grafo, codigo):
    for aeropuerto in grafo.L.keys():
        if aeropuerto.code == codigo:
            return aeropuerto
    return None

############### GRAFO  ###############

def plot_graph(graph: Graph):
    G = nx.Graph()
    
    for airport in graph.L: #Añado los aeropuertos y su código
        G.add_node(airport.code, label=airport.name, pos=(airport.longitude, airport.latitude))

    for (u, v), cost in graph.cost_matrix.items(): #Añado las distancias
        G.add_edge(u.code, v.code, weight=cost)

    pos = {airport.code: (airport.longitude, airport.latitude) for airport in graph.L}

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{node}" for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            color='lightblue',
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Grafo de Vuelos",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()

#plot_graph(graph)

############### INTERFAZ ###############

from customtkinter import *
from CTkTable import CTkTable
from PIL import Image

app = CTk()
app.geometry("856x645")
app.resizable(0,0)
app.title("Viajes Aéreos")

set_appearance_mode("dark")

def conectividad_view():
    conectividad_frame = CTkFrame(master=app, fg_color="#fff", width=680, height=645, corner_radius=0)
    title_frame = CTkFrame(master=conectividad_frame, fg_color="transparent")
    title_frame.pack(anchor="n", fill="x", padx=27, pady=(35, 40))

    CTkLabel(master=title_frame, text="Conectividad Grafo", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")
    insert_icon_img_data = Image.open("insert_btn.png")
    insert_icon_img = CTkImage(dark_image=insert_icon_img_data, light_image=insert_icon_img_data, size=(20, 20))

    conectividad_text = CTkTextbox(master=conectividad_frame, font=("Arial", 18), height=260, width=550, fg_color="#adb5bd", corner_radius=20)  # Cambié el height y width
    conectividad_text.configure(state="disabled")

    def on_conectividad_click():
        conectividad_text.configure(state="normal")
        conectividad_text.delete("0.0", "end")  # Limpia el contenido actual
        parrafo_peso = graph.grafo_conexo_texto()
        
        conectividad_text.insert("0.0", parrafo_peso)
        conectividad_text.configure(state="disabled")  # Deshabilitar el textbox
    
    peso_arbol2_btn = CTkButton(master=conectividad_frame, text="CONECTIVIDAD", image=insert_icon_img, fg_color="#008CFF", hover_color="#00A3FF", font=("Arial Black", 14),
                                corner_radius=10, height=80, width=180, text_color="#FFFFFF", border_width=2, border_color="#006BBF", command=on_conectividad_click)
    peso_arbol2_btn.pack(pady=30)  # Mantengo el espacio alrededor del botón

    conectividad_text.pack(pady=10, padx=50)  # Agregué padding horizontal para centrarlo
    conectividad_frame.pack(side="left", fill="both", expand=True)
    conectividad_frame.pack_propagate(0)
    
def peso_arbol_view():
    peso_arbol_frame = CTkFrame(master=app, fg_color="#fff", width=680, height=645, corner_radius=0)
    title_frame = CTkFrame(master=peso_arbol_frame, fg_color="transparent")
    title_frame.pack(anchor="n", fill="x", padx=27, pady=(35, 40))

    CTkLabel(master=title_frame, text="Peso Árbol", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")
    
    recorrido_icon_data = Image.open("recorrido_icon.png")
    recorrido_icon = CTkImage(dark_image=recorrido_icon_data, light_image=recorrido_icon_data, size=(20, 20))

    # Ajustando el TextBox para que sea más pequeño y centralizado
    peso_arbol_text = CTkTextbox(master=peso_arbol_frame, font=("Arial", 18), height=220, width=550, fg_color="#adb5bd", corner_radius=20)  # Cambié el height y width
    peso_arbol_text.configure(state="disabled")

    def on_pesoArbol_click():
        peso_arbol_text.configure(state="normal")
        peso_arbol_text.delete("0.0", "end")  # Limpia el contenido actual
        parrafo_peso = graph.kruskal_costos_porArbol_texto()
        
        peso_arbol_text.insert("0.0", parrafo_peso)
        peso_arbol_text.configure(state="disabled")  # Deshabilitar el textbox
    
    peso_arbol2_btn = CTkButton(master=peso_arbol_frame, text="PESO ÁRBOL", image=recorrido_icon, fg_color="#008CFF", hover_color="#00A3FF", font=("Arial Black", 14),
                                corner_radius=10, height=80, width=180, text_color="#FFFFFF", border_width=2, border_color="#006BBF", command=on_pesoArbol_click)
    peso_arbol2_btn.pack(pady=30)  # Mantengo el espacio alrededor del botón

    peso_arbol_text.pack(pady=10, padx=50)  # Agregué padding horizontal para centrarlo

    peso_arbol_frame.pack(side="left", fill="both", expand=True)
    peso_arbol_frame.pack_propagate(0)

def buscar_1vertice_view():
    buscar_frame = CTkFrame(master=app, fg_color="#fff", width=680, height=645, corner_radius=0)
    title_frame = CTkFrame(master=buscar_frame, fg_color="transparent")
    title_frame.pack(anchor="n", fill="x", padx=27, pady=(35, 0))

    CTkLabel(master=title_frame, text="Buscar Aeropuerto", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")

    buscar_aeropuerto_frame = CTkFrame(master=buscar_frame, height=50, width=32, fg_color="#F0F0F0")
    buscar_aeropuerto_frame.pack(fill="x", pady=(45, 0), padx=20)

    buscar_aeropuerto_entry = CTkEntry(master=buscar_aeropuerto_frame, width=400, placeholder_text="Escriba el código del aeropuerto", border_color="#333333", border_width=2)
    buscar_aeropuerto_entry.pack(side="left", padx=(13, 0), pady=15)

    buscar_icon_img_data = Image.open("buscar_btn.png") 
    buscar_icon_img = CTkImage(dark_image=buscar_icon_img_data, light_image=buscar_icon_img_data)

    # Frame contenedor para el primer TextBox
    contenedor1_frame = CTkFrame(master=buscar_frame, fg_color="#fff")
    contenedor1_frame.pack(fill="x", pady=(20, 0), padx=20)

    # Primer TextBox
    buscar_1vertice_text = CTkTextbox(master=contenedor1_frame, font=("Arial", 18), height=120, width=420, fg_color="#adb5bd", corner_radius=20)
    buscar_1vertice_text.configure(state="disabled")
    buscar_1vertice_text.pack(fill="x")

    # Subtítulo "Caminos Mínimos a partir del vértice"
    subtitle_frame = CTkFrame(master=buscar_frame, fg_color="transparent")
    subtitle_frame.pack(fill="x", padx=20, pady=(20, 0))  # Espacio entre el primer frame y el subtítulo
    CTkLabel(master=subtitle_frame, text="Caminos mínimos a partir del vértice", font=("Arial Black", 20), text_color="#333333").pack(anchor="nw", side="left")

    # Frame contenedor para el segundo TextBox
    contenedor2_frame = CTkFrame(master=buscar_frame, fg_color="#fff")
    contenedor2_frame.pack(fill="x", pady=(20, 0), padx=20)

    # Segundo TextBox
    buscar_caminominimo_text = CTkTextbox(master=contenedor2_frame, font=("Arial", 18), height=190, width=420, fg_color="#adb5bd", corner_radius=20)
    buscar_caminominimo_text.configure(state="disabled")
    buscar_caminominimo_text.pack(fill="x")

    def on_1vertice_click():
        aeropuerto1 = obtener_por_codigo(graph, buscar_aeropuerto_entry.get())
        if aeropuerto1 is None:
            buscar_1vertice_text.configure(state="normal")
            buscar_1vertice_text.delete("0.0", "end")  # Limpia el contenido actual
            buscar_1vertice_text.insert("0.0", "Aeropuerto no encontrado.")
            buscar_1vertice_text.configure(state="disabled")

            buscar_caminominimo_text.configure(state="normal")
            buscar_caminominimo_text.delete("0.0", "end")  # Limpia el contenido actual
            buscar_caminominimo_text.insert("0.0", "")
            buscar_caminominimo_text.configure(state="disabled")
            return
        
        str1, str2 = graph.caminos_desde(aeropuerto1)

        buscar_1vertice_text.configure(state="normal")
        buscar_1vertice_text.delete("0.0", "end")  # Limpia el contenido actual
        buscar_1vertice_text.insert("0.0", str1)
        buscar_1vertice_text.configure(state="disabled")  # Deshabilitar el textbox

        buscar_caminominimo_text.configure(state="normal")
        buscar_caminominimo_text.delete("0.0", "end")  # Limpia el contenido actual
        buscar_caminominimo_text.insert("0.0", str2)
        buscar_caminominimo_text.configure(state="disabled")  # Deshabilitar el textbox
        
    buscar_aeropuerto_btn = CTkButton(master=buscar_aeropuerto_frame, text="BUSCAR", fg_color="#008CFF", hover_color="#00A3FF", font=("Arial Black", 14),
            corner_radius=10, height=50, width=180, text_color="#FFFFFF", image=buscar_icon_img, border_width=2, border_color="#006BBF", command=on_1vertice_click)
    buscar_aeropuerto_btn.pack(side="left", padx=(20, 0), pady=15)

    buscar_frame.pack(side="left", fill="both", expand=True)
    buscar_frame.pack_propagate(0)
    
def buscar_2vertices_view():
    buscar_2vertices_frame = CTkFrame(master=app, fg_color="#fff", width=680, height=645, corner_radius=0)
    title_frame = CTkFrame(master=buscar_2vertices_frame, fg_color="transparent")
    title_frame.pack(anchor="n", fill="x", padx=27, pady=(35, 0))

    CTkLabel(master=title_frame, text="Camino mínimo entre dos vértices", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")

    buscar_vertices_frame = CTkFrame(master=buscar_2vertices_frame, height=50, fg_color="#F0F0F0")
    buscar_vertices_frame.pack(fill="x", pady=(45, 0), padx=20)

    # Configurar el grid en el frame para los CTkEntry y el botón
    buscar_vertices_frame.grid_columnconfigure(0, weight=1)
    buscar_vertices_frame.grid_columnconfigure(1, weight=1)
    buscar_vertices_frame.grid_columnconfigure(2, weight=0)

    # Crear los CTkEntry
    buscar_primer_aeropuerto_entry = CTkEntry(master=buscar_vertices_frame, width=380, placeholder_text="Escriba el primer código de aeropuerto", border_color="#333333", border_width=2)
    buscar_primer_aeropuerto_entry.grid(row=0, column=0, padx=(13, 5), pady=15, sticky="ew")
    
    buscar_segundo_aeropuerto_entry = CTkEntry(master=buscar_vertices_frame, width=380, placeholder_text="Escriba el segundo código de aeropuerto", border_color="#333333", border_width=2)
    buscar_segundo_aeropuerto_entry.grid(row=0, column=1, padx=(5, 5), pady=15, sticky="ew")

    buscar_metrica_icon_img_data = Image.open("buscar_btn.png")
    buscar_metrica_icon_img = CTkImage(dark_image=buscar_metrica_icon_img_data, light_image=buscar_metrica_icon_img_data)

    info_text = CTkTextbox(master=buscar_2vertices_frame, font=("Arial", 18), height=150, width=550, fg_color="#adb5bd", corner_radius=20)
    info_text.configure(state="disabled")
    info_text.pack(pady=50, padx=50)

    def on_2vertices_click():
        aeropuerto1 = obtener_por_codigo(graph, buscar_primer_aeropuerto_entry.get())
        aeropuerto2 = obtener_por_codigo(graph, buscar_segundo_aeropuerto_entry.get())

        if aeropuerto1 is None or aeropuerto2 is None:
            if aeropuerto1 is None:
                buscar_primer_aeropuerto_entry.delete(0, "end")
                buscar_primer_aeropuerto_entry.insert(0, "No encontrado")
            if aeropuerto2 is None:
                buscar_segundo_aeropuerto_entry.delete(0, "end")
                buscar_segundo_aeropuerto_entry.insert(0, "No encontrado")
            return  # Salir si no se encuentra alguno de los aeropuertos
        
        camino_minimo = graph.dijkstra_dosVertices(aeropuerto1, aeropuerto2)

        if not camino_minimo:
            info_text.configure(state="normal")
            info_text.delete("0.0", "end")  # Limpia el contenido actual
            info_text.insert("0.0", "No se encontró un camino del origen al destino.")
            info_text.configure(state="disabled")  # Deshabilitar el textbox
        else:
            info_text.configure(state="normal")
            info_text.delete("0.0", "end")  # Limpia el contenido actual
            info_text.insert("0.0", "Mapa generado con éxito.")
            info_text.configure(state="disabled")  # Deshabilitar el textbox
            graph.mostrar_camino_en_mapa(camino_minimo)
    
    buscar_2vertices_btn = CTkButton(master=buscar_vertices_frame, text="BUSCAR", image=buscar_metrica_icon_img, fg_color="#008CFF", hover_color="#00A3FF", font=("Arial Black", 14),
                                     corner_radius=10, height=50, width=180, text_color="#FFFFFF", border_width=2, border_color="#006BBF", command=on_2vertices_click)
    buscar_2vertices_btn.grid(row=0, column=2, padx=(15, 10), pady=15)

    buscar_2vertices_frame.pack(side="left", fill="both", expand=True)
    buscar_2vertices_frame.pack_propagate(0)

def perfil_view():
    start_frame = CTkFrame(master=app, fg_color="#fff",  width=680, height=645, corner_radius=0)
    title_frame = CTkFrame(master=start_frame, fg_color="transparent")
    title_frame.pack(anchor="n", fill="x",  padx=27, pady=(35, 0))

    title_frame.pack(anchor="n", fill="x",  padx=27, pady=(35, 0))
    CTkLabel(master=title_frame, text="DataSet Flights Final", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")

    descripcion_frame = CTkFrame(master=start_frame, height=650, width=50, fg_color="transparent", corner_radius=20)
    descripcion_frame.pack(fill="x", pady=(70, 0), padx=60)

    descripcion_text = CTkTextbox(master=descripcion_frame, font=("Arial", 18), height=230, width=60, fg_color="#adb5bd", corner_radius=20)
    parrafo = """¡Bienvenido al Sistema de Gestión de Viajes! Aquí podrás
administrar tus viajes y obtener información detallada sobre
diferentes aeropuertos del mundo de manera fácil y rápida. Ya
sea que desees consultar la información de cada aeropuerto,
ver las rutas entre ellos, los vuelos disponibles o la distancia
entre los destinos, este programa te brinda todas las
herramientas necesarias para planificar y organizar tus viajes
de manera eficiente. ¡Comencemos!"""
    descripcion_text.insert("0.0", parrafo)
    descripcion_text.configure(state="disabled")
    descripcion_text.pack(fill="both", expand=True)

    autores_frame = CTkFrame(master=start_frame, fg_color="transparent")
    autores_frame.pack(anchor="n", fill="x",  padx=27, pady=(110, 0))

    autora_metrica = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
    autora_metrica.grid_propagate(0)
    autora_metrica.pack(side="left")

    mujer_img_data = Image.open("mujer.png")
    mujer_img = CTkImage(light_image=mujer_img_data, dark_image=mujer_img_data, size=(43, 43))

    CTkLabel(master=autora_metrica, image=mujer_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

    CTkLabel(master=autora_metrica, text="Kesly Rodríguez", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
    CTkLabel(master=autora_metrica, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco


    autor1_metric = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
    autor1_metric.grid_propagate(0)
    autor1_metric.pack(side="left", expand=True, anchor="center")

    hombre2_img_data = Image.open("hombre2.png")
    hombre2_img = CTkImage(light_image=hombre2_img_data, dark_image=hombre2_img_data, size=(43, 43))
    CTkLabel(master=autor1_metric, image=hombre2_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

    CTkLabel(master=autor1_metric, text="Santiago V.", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
    CTkLabel(master=autor1_metric, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco

    autor2_metric = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
    autor2_metric.grid_propagate(0)
    autor2_metric.pack(side="right")

    hombre1_img_data = Image.open("hombre1.png")
    hombre1_img = CTkImage(light_image=hombre1_img_data, dark_image=hombre1_img_data, size=(43, 43))

    CTkLabel(master=autor2_metric, image=hombre1_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

    CTkLabel(master=autor2_metric, text="Joshua Lobo", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
    CTkLabel(master=autor2_metric, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco


    start_frame.pack(side="left", fill = "both", expand = True)
    start_frame.pack_propagate(0)

sidebar_frame = CTkFrame(master=app, fg_color="#ffb200",  width=176, height=650, corner_radius=0)
sidebar_frame.pack_propagate(0)
sidebar_frame.pack(fill="y", anchor="w", side="left")

logo_img_data = Image.open("logo_aeropuerto.png")
logo_img = CTkImage(dark_image=logo_img_data, light_image=logo_img_data, size=(120, 120))    

CTkLabel(master=sidebar_frame, text="", image=logo_img).pack(pady=(38, 0), anchor="center")

# Función para cambiar el color del botón al hacer clic
current_button = None
def on_button_click(button):
    global current_button
    if current_button:
        current_button.configure(fg_color="transparent")  # Color inicial del botón
    button.configure(fg_color="#333333")  # Color del hover_color
    current_button = button

def delete_pages():
    for frame in app.winfo_children():
        if frame != sidebar_frame:
            frame.destroy()

def combined_function1():
    delete_pages()
    conectividad_view()
    on_button_click(conectividad_btn)
    
def combined_function2():
    delete_pages()
    peso_arbol_view()
    on_button_click(peso_arbol_btn)

def combined_function3():
    delete_pages()
    buscar_1vertice_view()
    on_button_click(buscar_1vertice_btn)

def combined_function4():
    delete_pages()
    buscar_2vertices_view()
    on_button_click(buscar_2vertices_btn)
    
def combined_function5():
    delete_pages()
    perfil_view()
    on_button_click(perfil_btn)

def combined_function6():
    graph.mostrar_mapa()
    on_button_click(mapa_btn)

conectividad_img_data = Image.open("insertar.png")
conectividad_img = CTkImage(dark_image=conectividad_img_data, light_image=conectividad_img_data)

conectividad_btn = CTkButton(master=sidebar_frame, image=conectividad_img, text="Conectividad", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command= combined_function1)
conectividad_btn.pack(anchor="center", ipady=5, pady=(60, 0))

peso_arbol_img_data = Image.open("eliminar.png")
peso_arbol_img = CTkImage(dark_image=peso_arbol_img_data, light_image=peso_arbol_img_data)

peso_arbol_btn = CTkButton(master=sidebar_frame, image=peso_arbol_img, text="Peso árbol", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command= combined_function2)
peso_arbol_btn.pack(anchor="center", ipady=5, pady=(16, 0))


buscar_1vertice_img_data = Image.open("buscar.png")
buscar_1vertice_img = CTkImage(dark_image=buscar_1vertice_img_data, light_image=buscar_1vertice_img_data)

buscar_1vertice_btn = CTkButton(master=sidebar_frame, image=buscar_1vertice_img, text="Info. un vértice", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command= combined_function3)
buscar_1vertice_btn.pack(anchor="center", ipady=5, pady=(16, 0))

buscar_2vertices_img_data = Image.open("buscar_metrica.png")
buscar_2vertices_img = CTkImage(dark_image=buscar_2vertices_img_data, light_image=buscar_2vertices_img_data)

buscar_2vertices_btn = CTkButton(master=sidebar_frame, image=buscar_2vertices_img, text="Info. dos vértices", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command= combined_function4)
buscar_2vertices_btn.pack(anchor="center", ipady=5, pady=(16, 0))

mapa_img_data = Image.open("mapa.png")
mapa_img = CTkImage(dark_image=mapa_img_data, light_image=mapa_img_data)

mapa_btn = CTkButton(master=sidebar_frame, image=mapa_img, text="Mapa", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command= combined_function6)
mapa_btn.pack(anchor="center", ipady=5, pady=(16, 0))

person_img_data = Image.open("person_icon.png")
person_img = CTkImage(dark_image=person_img_data, light_image=person_img_data)
perfil_btn = CTkButton(master=sidebar_frame, image=person_img, text="Perfil", fg_color="transparent", font=("Arial Bold", 14), hover_color="#333333", anchor="w", command=combined_function5)
perfil_btn.pack(anchor="center", ipady=5, pady=(100, 0))

start_frame = CTkFrame(master=app, fg_color="#fff",  width=680, height=645, corner_radius=0)
title_frame = CTkFrame(master=start_frame, fg_color="transparent")
title_frame.pack(anchor="n", fill="x",  padx=27, pady=(35, 0))

CTkLabel(master=title_frame, text="DataSet Flights Final", font=("Arial Black", 30), text_color="#333333").pack(anchor="nw", side="left")

descripcion_frame = CTkFrame(master=start_frame, height=650, width=50, fg_color="transparent", corner_radius=20)
descripcion_frame.pack(fill="x", pady=(70, 0), padx=60)

descripcion_text = CTkTextbox(master=descripcion_frame, font=("Arial", 18), height=230, width=60, fg_color="#adb5bd", corner_radius=20)
parrafo = """¡Bienvenido al Sistema de Gestión de Viajes! Aquí podrás
administrar tus viajes y obtener información detallada sobre
diferentes aeropuertos del mundo de manera fácil y rápida. Ya
sea que desees consultar la información de cada aeropuerto,
ver las rutas entre ellos, los vuelos disponibles o la distancia
entre los destinos, este programa te brinda todas las
herramientas necesarias para planificar y organizar tus viajes
de manera eficiente. ¡Comencemos!"""
descripcion_text.insert("0.0", parrafo)
descripcion_text.configure(state="disabled")
descripcion_text.pack(fill="both", expand=True)

autores_frame = CTkFrame(master=start_frame, fg_color="transparent")
autores_frame.pack(anchor="n", fill="x",  padx=27, pady=(110, 0))

autora_metrica = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
autora_metrica.grid_propagate(0)
autora_metrica.pack(side="left")

mujer_img_data = Image.open("mujer.png")
mujer_img = CTkImage(light_image=mujer_img_data, dark_image=mujer_img_data, size=(43, 43))

CTkLabel(master=autora_metrica, image=mujer_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

CTkLabel(master=autora_metrica, text="Kesly Rodríguez", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
CTkLabel(master=autora_metrica, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco

autor1_metric = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
autor1_metric.grid_propagate(0)
autor1_metric.pack(side="left", expand=True, anchor="center")

hombre2_img_data = Image.open("hombre2.png")
hombre2_img = CTkImage(light_image=hombre2_img_data, dark_image=hombre2_img_data, size=(43, 43))
CTkLabel(master=autor1_metric, image=hombre2_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

CTkLabel(master=autor1_metric, text="Santiago V.", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
CTkLabel(master=autor1_metric, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco

autor2_metric = CTkFrame(master=autores_frame, fg_color="#333333", width=200, height=60)  # Cambiado a negro
autor2_metric.grid_propagate(0)
autor2_metric.pack(side="right")

hombre1_img_data = Image.open("hombre1.png")
hombre1_img = CTkImage(light_image=hombre1_img_data, dark_image=hombre1_img_data, size=(43, 43))

CTkLabel(master=autor2_metric, image=hombre1_img, text="").grid(row=0, column=0, rowspan=2, padx=(12,5), pady=10)

CTkLabel(master=autor2_metric, text="Joshua Lobo", text_color="#FFFFFF", font=("Arial Black", 15)).grid(row=0, column=1, sticky="sw")  # Cambiado a texto blanco
CTkLabel(master=autor2_metric, text="Ing. Sistemas", text_color="#FFFFFF", font=("Arial Black", 15), justify="left").grid(row=1, column=1, sticky="nw", pady=(0,10))  # Cambiado a texto blanco

start_frame.pack(side="left", fill = "both", expand = True)
start_frame.pack_propagate(0)

app.mainloop()


