import networkx as nx
from typing import Dict, List, Literal, Tuple, Set
from collections import defaultdict

from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df

class EvacuationPolicy:
    """
    Tu implementación de la política de evacuación.
    Esta es la clase que necesitas implementar para resolver el problema de evacuación.
    """
    
    def __init__(self):
        """Inicializa tu política de evacuación"""
        self.policy_type = "policy_1"  # Política por defecto
        
    def set_policy(self, policy_type: Literal["policy_1", "policy_2", "policy_3", "policy_4"]):
        """
        Selecciona la política a utilizar
        Args:
            policy_type: Tipo de política a utilizar
                - "policy_1": Política básica sin uso de proxies
                - "policy_2": Política usando proxies y sus descripciones
                - "policy_3": Política usando datos de simulaciones previas
                - "policy_4": Política personalizada
        """
        self.policy_type = policy_type
    
    def plan_evacuation(self, city: CityGraph, proxy_data: ProxyData, 
                       max_resources: int) -> PolicyResult:
        """
        Planifica la ruta de evacuación y la asignación de recursos.
        
        Args:
            city: El layout de la ciudad
                 - city.graph: Grafo NetworkX con el layout de la ciudad
                 - city.starting_node: Tu posición inicial
                 - city.extraction_nodes: Lista de puntos de extracción posibles
                 
            proxy_data: Información sobre el ambiente
                 - proxy_data.node_data[node_id]: Dict con indicadores de nodos
                 - proxy_data.edge_data[(node1,node2)]: Dict con indicadores de aristas
                 
            max_resources: Máximo total de recursos que puedes asignar
            
        Returns:
            PolicyResult con:
            - path: List[int] - Lista de IDs de nodos formando tu ruta de evacuación
            - resources: Dict[str, int] - Cuántos recursos de cada tipo llevar:
                       {'explosives': x, 'ammo': y, 'radiation_suits': z}
                       donde x + y + z <= max_resources
        """
        # print(f'City graph: {city.graph} \n')
        # print(f'City starting_node: {city.starting_node}\n')
        # print(f'City extraction_nodes: {city.extraction_nodes}\n')
        # print(f'Proxy node_data: {proxy_data.node_data} \n \n')
        # print(f'Proxy edge_data: {proxy_data.edge_data} \n \n')
        # print(f'Max Resources: {max_resources} \n \n')
        
        
        self.policy_type = "policy_4" # TODO: Cambiar a "policy_2" para probar la política 2, y asi sucesivamente
        
        if self.policy_type == "policy_1":
            return self._policy_1(city, max_resources)
        elif self.policy_type == "policy_2":
            return self._policy_2(city, proxy_data, max_resources)
        elif self.policy_type == "policy_3":
            return self._policy_3(city, proxy_data, max_resources)
        else:  # policy_4
            return self._policy_4(city, proxy_data, max_resources)
    
    def _policy_1(self, city: CityGraph, max_resources: int) -> PolicyResult:
        # Obtenemos el grafo de la ciudad, el nodo de inicio y la lista de nodos de extracción
        graph = city.graph
        start = city.starting_node
        extraction_nodes = city.extraction_nodes

        # Inicializamos las variables para almacenar el mejor camino y la longitud mínima encontrada
        best_path = None
        min_length = float('inf')  # Se asigna infinito para que cualquier camino válido sea menor

        # Iteramos sobre cada nodo de extracción para encontrar el camino más corto desde el nodo de inicio
        for target in extraction_nodes:
            try:
                # Calculamos la longitud del camino más corto utilizando el peso 'weight'
                path_length = nx.shortest_path_length(graph, start, target, weight='weight')
                # Si la longitud del camino actual es menor que la mínima encontrada, actualizamos las variables
                if path_length < min_length:
                    min_length = path_length
                    # Obtenemos y almacenamos el camino más corto (lista de nodos) desde 'start' hasta 'target'
                    best_path = nx.shortest_path(graph, start, target, weight='weight')
            except nx.NetworkXNoPath:
                # Si no hay camino entre 'start' y 'target', se ignora este nodo y se continúa con el siguiente
                continue

        # Si no se encontró ningún camino hacia los nodos de extracción, se asigna el camino como el nodo de inicio únicamente
        if best_path is None:
            best_path = [start]

        # Distribución de recursos:
        # Se dividen de manera equitativa max_resources entre tres tipos: explosivos, munición y trajes de radiación
        explosives = max_resources // 3
        ammo = max_resources // 3
        suits = max_resources // 3
        # Calculamos el remanente de recursos que no se distribuyó equitativamente
        remaining = max_resources - (explosives + ammo + suits)

        # Se reparte el remanente, asignando uno extra a cada recurso en orden hasta agotarlo
        if remaining > 0:
            explosives += 1
            remaining -= 1
        if remaining > 0:
            ammo += 1
            remaining -= 1
        if remaining > 0:
            suits += 1

        # Se crea un diccionario con la cantidad final asignada a cada recurso
        resources = {
            'explosives': explosives,
            'ammo': ammo,
            'radiation_suits': suits
        }
        
        return PolicyResult(best_path, resources)
    

        # Se retorna un objeto PolicyResult que contiene la mejor ruta encontrada y la distribución de recursos
    def _policy_2(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        # Se crea una copia del grafo para no modificar el original
        adjusted_graph = city.graph.copy()

        # --- Cálculo del riesgo en cada nodo (Indicadores de Ubicaciones) ---
        # Indicadores evaluados:
        # - Actividad sísmica (seismic_activity): 0 = estable, 1 = riesgo alto.
        # - Lecturas de radiación (radiation_readings): 0 = radiación de fondo, 1 = niveles letales.
        # - Densidad poblacional (population_density): 0 = desierta, 1 = alta concentración.
        # - Llamadas de emergencia (emergency_calls): 0 = sin señales, 1 = señales extremas.
        # - Lecturas térmicas (thermal_readings): 0 = temperatura normal, 1 = actividad máxima.
        # - Fuerza de señal (signal_strength): 1 = conectividad perfecta, 0 = sin comunicación (se usa 1 - valor).
        # - Integridad estructural (structural_integrity): 1 = estructuras estables, 0 = inestabilidad (se usa 1 - valor).
        node_risks = {}
        for node in adjusted_graph.nodes():
            node_data = proxy_data.node_data.get(node, {})
            seismic_activity     = node_data.get('seismic_activity', 0)
            radiation_readings   = node_data.get('radiation_readings', 0)
            population_density   = node_data.get('population_density', 0)
            emergency_calls      = node_data.get('emergency_calls', 0)
            thermal_readings     = node_data.get('thermal_readings', 0)
            signal_strength      = node_data.get('signal_strength', 1)  # Conectividad buena por defecto
            structural_integrity = node_data.get('structural_integrity', 1)  # Estabilidad por defecto
            
            # Se suma el riesgo: los indicadores desfavorables se suman directamente; para señal e integridad se usa (1 - valor)
            risk = (seismic_activity + radiation_readings + population_density +
                    emergency_calls + thermal_readings + (1 - signal_strength) +
                    (1 - structural_integrity))
            node_risks[node] = risk

        # --- Ajuste de pesos en las aristas (Indicadores de Rutas) ---
        # Indicadores evaluados:
        # - Daño estructural (structural_damage): 0 = despejada, 1 = bloqueo total.
        # - Interferencia de señal (signal_interference): 0 = sin interferencia, 1 = total.
        # - Avistamientos de movimiento (movement_sightings): 0 = sin actividad, 1 = alta actividad.
        # - Densidad de escombros (debris_density): 0 = ruta despejada, 1 = mucha acumulación.
        # - Gradiente de peligro (hazard_gradient): 0 = condiciones uniformes, 1 = cambios bruscos.
        #
        # El riesgo de la arista se define como la suma de estos indicadores, al que se le suma
        # el riesgo promedio de los nodos adyacentes.
        for u, v in adjusted_graph.edges():
            edge_data = proxy_data.edge_data.get((u, v), {})
            structural_damage   = edge_data.get('structural_damage', 0)
            signal_interference = edge_data.get('signal_interference', 0)
            movement_sightings  = edge_data.get('movement_sightings', 0)
            debris_density      = edge_data.get('debris_density', 0)
            hazard_gradient     = edge_data.get('hazard_gradient', 0)
            edge_risk = (structural_damage + signal_interference +
                        movement_sightings + debris_density + hazard_gradient)
            
            total_risk = ((node_risks.get(u, 0) + node_risks.get(v, 0)) / 2) + edge_risk
            adjusted_graph[u][v]['weight'] = total_risk

        # --- Búsqueda del camino "más sweguro" ---
        # Se busca la ruta desde el nodo de inicio hasta uno de los nodos de extracción que minimice el riesgo total.
        best_path = None
        min_total_risk = float('inf')
        start = city.starting_node
        for target in city.extraction_nodes:
            try:
                path_risk = nx.shortest_path_length(adjusted_graph, start, target, weight='weight')
                if path_risk < min_total_risk:
                    min_total_risk = path_risk
                    best_path = nx.shortest_path(adjusted_graph, start, target, weight='weight')
            except nx.NetworkXNoPath:
                continue

        # Si no se encuentra ruta, se asigna el nodo de inicio como camino por defecto.
        if best_path is None:
            best_path = [start]

        # --- Determinación de recursos necesarios a lo largo del camino ---
        # Se combinan indicadores de aristas y nodos para asignar recursos de forma acorde al riesgo.
        # Para explosivos: se consideran el daño estructural en la arista y la vulnerabilidad de las estructuras en los nodos.
        # Para munición: se consideran los avistamientos de movimiento en la arista y las llamadas de emergencia en los nodos.
        # Para trajes de radiación: se evalúa la radiación en cada nodo.
        explosives_needed = 0
        ammo_needed = 0
        suits_needed = 0

        # Recorremos cada segmento del camino (entre dos nodos consecutivos)
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i+1]
            edge_data = proxy_data.edge_data.get((u, v), {})
            node_u_data = proxy_data.node_data.get(u, {})
            node_v_data = proxy_data.node_data.get(v, {})
            
            # --- Cálculo para Explosivos ---
            # Se toma en cuenta el daño estructural en la arista y la vulnerabilidad estructural en los nodos.
            edge_structural_damage = edge_data.get('structural_damage', 0)
            seismic_activity = (node_u_data.get('seismic_activity', 0) + node_v_data.get('seismic_activity', 0)) / 2
            # La vulnerabilidad se estima como (1 - structural_integrity) de cada nodo.
            node_vulnerability = ((1 - node_u_data.get('structural_integrity', 1)) +
                                (1 - node_v_data.get('structural_integrity', 1))) / 2
            # Si el promedio de ambos indicadores es igual o superior a 0.5, se requiere un explosivo.
            if (edge_structural_damage + node_vulnerability + seismic_activity)/3 >= 0.5:
                explosives_needed += 1

            # --- Cálculo para Munición ---
            # Se combinan los avistamientos de movimiento en la arista y las llamadas de emergencia en los nodos.
            edge_movement = edge_data.get('movement_sightings', 0)
            node_emergency = (node_u_data.get('emergency_calls', 0) + node_v_data.get('emergency_calls', 0)) / 2
            population_density = (node_u_data.get('population_density', 0) + node_v_data.get('population_density', 0)) / 2
            thermal_readings = (node_u_data.get('thermal_readings', 0) + node_v_data.get('thermal_readings', 0)) / 2

            if population_density >= 0.7:
                total = edge_movement + node_emergency + population_density
                if 0.4 < thermal_readings < 0.6:
                    total += thermal_readings
            else:
                total = edge_movement + node_emergency
                if 0.4 < thermal_readings < 0.6:
                    total += thermal_readings

            if total >= 0.5:
                ammo_needed += 1


        # Para los trajes de radiación se evalúa la radiación en cada nodo del camino.
        for node in best_path:
            node_data = proxy_data.node_data.get(node, {})
            radiation = node_data.get('radiation_readings', 0)
            if radiation >= 0.5:
                suits_needed += 1

        # --- Distribución de recursos disponibles (max_resources) ---
        # Se asignan primero los trajes de radiación, luego explosivos y finalmente munición,
        # sin exceder el total de recursos disponibles.
        suits = min(suits_needed, max_resources)
        remaining = max_resources - suits
        explosives = min(explosives_needed, remaining)
        remaining -= explosives
        ammo = min(ammo_needed, remaining)

        resources = {
            'explosives': explosives,
            'ammo': ammo,
            'radiation_suits': suits
        }

        return PolicyResult(best_path, resources)
    
    def _policy_3(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
            graph = city.graph
            start = city.starting_node
            extraction_nodes = city.extraction_nodes

            # 1. Filtrar nodos con radiación letal (excepto el nodo de inicio)
            nodes_to_remove = [
                node for node in graph.nodes 
                if node != start and proxy_data.node_data[node].get("radiation_readings", 0) >= 0.75
            ]
            filtered_graph = graph.copy()
            filtered_graph.remove_nodes_from(nodes_to_remove)

            # 2. Filtrar aristas peligrosas
            edges_to_remove = []
            for u, v in filtered_graph.edges():
                edge_key = (u, v) if (u, v) in proxy_data.edge_data else (v, u)
                if edge_key in proxy_data.edge_data:
                    hazard = proxy_data.edge_data[edge_key].get("hazard_gradient", 0)
                    if hazard >= 0.7:
                        edges_to_remove.append((u, v))
            filtered_graph.remove_edges_from(edges_to_remove)

            # 3. Validar nodos de extracción
            valid_extraction_nodes = [node for node in extraction_nodes if node in filtered_graph]

            # 4. Buscar camino más corto
            best_path = None
            min_length = float("inf")
            for target in valid_extraction_nodes:
                try:
                    path_length = nx.shortest_path_length(filtered_graph, start, target, weight="weight")
                    if path_length < min_length:
                        min_length = path_length
                        best_path = nx.shortest_path(filtered_graph, start, target, weight="weight")
                except nx.NetworkXNoPath:
                    continue

            # 5. Asignar ruta final (asegurar que el nodo de inicio siempre está incluido)
            final_path = best_path if best_path else [start]

            # 5. Manejar caso sin caminos válidos
            if not best_path:
                return PolicyResult([start], {"explosives": 0, "ammo": 0, "radiation_suits": 0})
            
            # Paso 4: Encontrar camino más corto usando Dijkstra
            best_path = None
            min_length = float("inf")
            for target in valid_extraction_nodes:
                try:
                    path_length = nx.shortest_path_length(filtered_graph, start, target, weight="weight")
                    if path_length < min_length:
                        min_length = path_length
                        best_path = nx.shortest_path(filtered_graph, start, target, weight="weight")
                except nx.NetworkXNoPath:
                    continue
            
            # Caso especial: No hay camino válido
            if not best_path:
                return PolicyResult([start], {"explosives": 0, "ammo": 0, "radiation_suits": 0})
            
            # Paso 5: Calcular recursos basados en máxima probabilidad
            explosives = 0
            suits = 0
            ammo = 0

            # Procesamiento de nodos
            for node in best_path:
                node_data = proxy_data.node_data[node]
                si = node_data.get("structural_integrity", 1)  # 1 = integridad perfecta
                sa = node_data.get("seismic_activity", 0)
                rad = node_data.get("radiation_readings", 0)
                pop = node_data.get("population_density", 0)
                emer = node_data.get("emergency_calls", 0)
                therm = node_data.get("thermal_readings", 0)

                # Cálculo de puntuaciones para cada recurso
                scores = {
                    "explosives": 0,
                    "radiation_suits": 0,
                    "ammo": 0
                }

                # Explosivos: Prioriza daño estructural severo (si <= 0.3) o actividad sísmica alta
                explosive_conditions = []
                if si <= 0.3:
                    explosive_conditions.append(1 - si)  # Invertir escala (mayor daño -> mayor puntuación)
                if 0.1 < si <= 0.3 and 0.3 < sa < 1:
                    explosive_conditions.append(sa)
                if explosive_conditions:
                    scores["explosives"] = max(explosive_conditions)

                # Trajes: Radiación cercana a niveles letales (0.3 < rad < 0.75)
                if 0.3 < rad < 0.75:
                    scores["radiation_suits"] = (rad - 0.3) / 0.45  # Escalado lineal a [0, 1]

                # Munición: Alta densidad poblacional/emergencias o actividad térmica moderada
                ammo_conditions = []
                if pop >= 0.6 or emer >= 0.6:
                    ammo_conditions.append(max(pop, emer))
                if 0.4 <= therm < 0.6:
                    ammo_conditions.append(therm)
                if ammo_conditions:
                    scores["ammo"] = max(ammo_conditions)

                # Seleccionar recurso con máxima puntuación
                max_score = max(scores.values())
                if max_score > 0:
                    selected = [k for k, v in scores.items() if v == max_score][0]  # En caso de empate, selecciona el primero
                    if selected == "explosives":
                        explosives += 1
                    elif selected == "radiation_suits":
                        suits += 1
                    else:
                        ammo += 1

            # Procesamiento de aristas
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                edge_key = (u, v) if (u, v) in proxy_data.edge_data else (v, u)
                if edge_key not in proxy_data.edge_data:
                    continue

                edge_data = proxy_data.edge_data[edge_key]
                sd = edge_data.get("structural_damage", 0)
                dd = edge_data.get("debris_density", 0)
                sig = edge_data.get("signal_interference", 0)

                # Comparar daño estructural vs interferencia
                explosive_score = max(sd, dd) if (sd >= 0.6 or dd >= 0.6) else 0
                suit_score = sig if sig >= 0.6 else 0

                if explosive_score > suit_score:
                    explosives += 1
                elif suit_score > explosive_score:
                    suits += 1

            return PolicyResult(final_path, {
                "explosives": min(explosives, max_resources),
                "ammo": min(ammo, max_resources),
                "radiation_suits": min(suits, max_resources),
            })
    
    def _policy_4(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 4: Estrategia con distribución inteligente de recursos.
        Basada en la política 3, pero mejora la distribución de recursos
        manteniendo las prioridades detectadas al analizar el camino.
        """
        # Definir umbrales de seguridad directamente en lugar de llamar a una función
        SAFETY_THRESHOLDS = {
            'radiation_critical': 0.6,
            'radiation_warning': 0.5,
            'population': 0.6,
            'emergency': 0.5,
            'thermal': (0.5, 0.6),
            'seismic': 0.5,
            'structural': 0.6,  # Umbral ajustado
            'movement': 0.45
        }
        
        graph = city.graph.copy()
        
        # 1. Calcular riesgos en nodos
        node_risks, forced_radiation_nodes = self._compute_node_risks(graph, proxy_data, SAFETY_THRESHOLDS)
        
        # 2. Ajustar riesgos en aristas
        self._adjust_edge_risks(graph, node_risks, proxy_data, SAFETY_THRESHOLDS, forced_radiation_nodes)
        
        # 3. Encontrar mejor camino
        best_path = self._find_optimal_path(graph, city, proxy_data, SAFETY_THRESHOLDS)
        
        # 4. Calcular necesidades de recursos
        resource_needs, critical_radiation_nodes = self._calculate_resource_needs(best_path, proxy_data, SAFETY_THRESHOLDS)
        
        # 5. Asignar recursos con distribución inteligente
        final_resources = self._distribute_resources_intelligently(
            best_path, resource_needs, critical_radiation_nodes, proxy_data, max_resources, SAFETY_THRESHOLDS
        )
        
        return PolicyResult(best_path, final_resources)

    def _distribute_resources_intelligently(self, path, resource_needs, critical_radiation_nodes, proxy_data, max_resources, thresholds):
        """
        Distribuye recursos de manera inteligente, priorizando según análisis de peligros en el camino,
        pero redistribuyendo cuando hay concentración excesiva en un tipo.
        """
        # Calcular puntuaciones de riesgo para cada tipo de recurso en el camino
        risk_scores = self._calculate_path_risk_scores(path, proxy_data, thresholds)
        
        # Normalizar las puntuaciones para obtener proporciones
        total_risk_score = sum(risk_scores.values())
        if total_risk_score == 0:
            # Si no hay riesgos detectados, distribuir equitativamente
            proportions = {
                'explosives': 0.33,
                'radiation_suits': 0.34,
                'ammo': 0.33
            }
        else:
            proportions = {
                resource: score / total_risk_score
                for resource, score in risk_scores.items()
            }
        
        # Asignar inicialmente según proporciones de riesgo
        initial_allocation = {
            resource: round(proportion * max_resources)
            for resource, proportion in proportions.items()
        }
        
        # Asegurar que todos los trajes de radiación críticos estén cubiertos
        needed_suits = len(critical_radiation_nodes)
        if initial_allocation['radiation_suits'] < needed_suits:
            deficit = needed_suits - initial_allocation['radiation_suits']
            # Intentar redistribuir desde otros recursos
            available_for_redistribution = sum(initial_allocation.values()) - needed_suits
            if available_for_redistribution >= 0:
                # Hay suficientes recursos para cubrir trajes + algo más
                initial_allocation['radiation_suits'] = needed_suits
                # Redistribuir el resto según proporciones de riesgo entre explosivos y munición
                remaining = max_resources - needed_suits
                other_risk_total = risk_scores['explosives'] + risk_scores['ammo']
                if other_risk_total > 0:
                    initial_allocation['explosives'] = round(remaining * (risk_scores['explosives'] / other_risk_total))
                    initial_allocation['ammo'] = remaining - initial_allocation['explosives']
                else:
                    # Si no hay otros riesgos, distribuir equitativamente
                    initial_allocation['explosives'] = remaining // 2
                    initial_allocation['ammo'] = remaining - initial_allocation['explosives']
        
        # Verificar mínimos esenciales para cada tipo basado en necesidades reales
        final_allocation = initial_allocation.copy()
        remaining_resources = max_resources - sum(final_allocation.values())
        
        # Asegurar mínimos esenciales si hay recursos disponibles
        for resource, min_needed in resource_needs.items():
            if min_needed > 0 and final_allocation[resource] == 0 and remaining_resources > 0:
                final_allocation[resource] = 1
                remaining_resources -= 1
        
        # Distribuir recursos restantes según prioridades
        if remaining_resources > 0:
            priority_order = sorted(
                ['explosives', 'ammo', 'radiation_suits'],
                key=lambda r: risk_scores[r],
                reverse=True
            )
            
            for resource in priority_order:
                if resource_needs[resource] > final_allocation[resource] and remaining_resources > 0:
                    additional = min(resource_needs[resource] - final_allocation[resource], remaining_resources)
                    final_allocation[resource] += additional
                    remaining_resources -= additional
        
        # Asegurar que la suma total no exceda max_resources (por redondeo)
        if sum(final_allocation.values()) > max_resources:
            excess = sum(final_allocation.values()) - max_resources
            for resource in sorted(['explosives', 'ammo', 'radiation_suits'], 
                                key=lambda r: risk_scores[r]):  # Reducir de menor a mayor prioridad
                if final_allocation[resource] > 0 and excess > 0:
                    reduction = min(final_allocation[resource], excess)
                    final_allocation[resource] -= reduction
                    excess -= reduction
        
        return final_allocation

    def _calculate_path_risk_scores(self, path, proxy_data, thresholds):
        """
        Calcula puntuaciones de riesgo para cada tipo de recurso basado en el análisis del camino.
        """
        risk_scores = {
            'explosives': 0,
            'radiation_suits': 0,
            'ammo': 0
        }
        
        # Analizar nodos del camino
        for node in path:
            data = proxy_data.node_data.get(node, {})
            
            # Riesgo de radiación
            radiation = data.get('radiation_readings', 0)
            if radiation > thresholds['radiation_critical']:
                risk_scores['radiation_suits'] += 100  # Prioridad crítica
            elif radiation > thresholds['radiation_warning']:
                risk_scores['radiation_suits'] += 40
            
            # Riesgo estructural/sísmico
            structural_risk = (1 - data.get('structural_integrity', 1))
            seismic_activity = data.get('seismic_activity', 0)
            if (seismic_activity + structural_risk) / 2 >= thresholds['structural']:
                risk_scores['explosives'] += 50
            else:
                risk_scores['explosives'] += ((seismic_activity + structural_risk) / 2) * 30
            
            # Riesgo de población y emergencias
            population_risk = data.get('population_density', 0)
            emergency_risk = data.get('emergency_calls', 0)
            if population_risk > thresholds['population'] or emergency_risk > thresholds['emergency']:
                risk_scores['ammo'] += 60
            else:
                risk_scores['ammo'] += (population_risk + emergency_risk) * 20
        
        # Analizar aristas del camino
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_data = proxy_data.edge_data.get((u, v), {})
            
            # Riesgo estructural en aristas
            structural_damage = edge_data.get('structural_damage', 0)
            if structural_damage > thresholds['structural']:
                risk_scores['explosives'] += 40
            else:
                risk_scores['explosives'] += structural_damage * 25
            
            # Riesgo de movimiento hostil
            movement = edge_data.get('movement_sightings', 0)
            if movement > thresholds['movement']:
                risk_scores['ammo'] += 45
            else:
                risk_scores['ammo'] += movement * 30
        
        return risk_scores
    
    def _compute_node_risks(self, graph, proxy_data, thresholds) -> Tuple[Dict, Set]:
        node_risks = {}
        forced_radiation_nodes = set()
        
        for node in graph.nodes():
            data = proxy_data.node_data.get(node, {})
            radiation = data.get('radiation_readings', 0)
            
            if radiation > thresholds['radiation_critical']:
                forced_radiation_nodes.add(node)
                node_risks[node] = 1000
            else:
                risk = (
                    data.get('seismic_activity', 0) * 2 +
                    (1 - data.get('structural_integrity', 1)) * 3 +
                    (data.get('population_density', 0)**2 * 4) +
                    data.get('emergency_calls', 0) * 2
                )
                if radiation > thresholds['radiation_warning']:
                    risk += radiation * 50
                node_risks[node] = risk
                
        return node_risks, forced_radiation_nodes

    def _adjust_edge_risks(self, graph, node_risks, proxy_data, thresholds, forced_nodes):
        for u, v in graph.edges():
            edge_data = proxy_data.edge_data.get((u, v), {})
            radiation_penalty = 500 if u in forced_nodes or v in forced_nodes else 0
            
            edge_risk = (
                edge_data.get('structural_damage', 0) * 3 +
                edge_data.get('movement_sightings', 0)**1.5 * 4 +
                edge_data.get('debris_density', 0) * 2 +
                radiation_penalty
            )
            
            graph[u][v]['weight'] = (node_risks[u] + node_risks[v]) * 0.4 + edge_risk

    def _find_optimal_path(self, graph, city, proxy_data, thresholds) -> List:
        best_path = []
        min_combined_risk = float('inf')
        
        for target in city.extraction_nodes:
            try:
                path = nx.shortest_path(graph, city.starting_node, target, weight='weight')
                radiation_risk = sum(
                    1000 for node in path
                    if proxy_data.node_data.get(node, {}).get('radiation_readings', 0) > thresholds['radiation_critical']
                )
                total_risk = nx.path_weight(graph, path, 'weight') + radiation_risk
                
                if total_risk < min_combined_risk:
                    min_combined_risk = total_risk
                    best_path = path
            except nx.NetworkXNoPath:
                continue
        
        return best_path if best_path else [city.starting_node]

    def _calculate_resource_needs(self, best_path, proxy_data, thresholds) -> Tuple[Dict, Set]:
        resource_needs = defaultdict(int)
        critical_radiation_nodes = set()
        
        # Analizar nodos del camino
        for node in best_path:
            data = proxy_data.node_data.get(node, {})
            
            if data.get('radiation_readings', 0) > thresholds['radiation_critical']:
                resource_needs['radiation_suits'] += 1
                critical_radiation_nodes.add(node)
                
            structural_risk = (1 - data.get('structural_integrity', 1))
            seismic_activity = data.get('seismic_activity', 0)
            if (seismic_activity + structural_risk) / 2 >= thresholds['structural']:
                resource_needs['explosives'] += 1
                
            if (data.get('population_density', 0) > thresholds['population'] or 
                data.get('emergency_calls', 0) > thresholds['emergency']):
                resource_needs['ammo'] += 1

        # Analizar aristas del camino
        for i in range(len(best_path)-1):
            u, v = best_path[i], best_path[i+1]
            edge_data = proxy_data.edge_data.get((u, v), {})
            
            if edge_data.get('structural_damage', 0) > thresholds['structural']:
                resource_needs['explosives'] += 1
                
            if edge_data.get('movement_sightings', 0) > thresholds['movement']:
                resource_needs['ammo'] += 1
                
        return resource_needs, critical_radiation_nodes

    def _allocate_resources(self, resource_needs, critical_radiation_nodes, max_resources) -> Dict:
        allocated = defaultdict(int)
        remaining = max_resources
        
        # Prioridad 1: Trajes de radiación
        needed_suits = len(critical_radiation_nodes)
        allocated['radiation_suits'] = min(needed_suits, remaining)
        remaining -= allocated['radiation_suits']
        
        if allocated['radiation_suits'] < needed_suits:
            deficit = needed_suits - allocated['radiation_suits']
            remaining = max(0, remaining - deficit * 2)
            
        # Prioridad 2: Explosivos
        allocated['explosives'] = min(resource_needs['explosives'], remaining)
        remaining -= allocated['explosives']
        
        # Prioridad 3: Munición
        allocated['ammo'] = min(resource_needs['ammo'], remaining)
        remaining -= allocated['ammo']
        
        # Garantizar mínimo de munición
        if allocated['ammo'] == 0 and resource_needs['ammo'] > 0 and remaining > 0:
            allocated['ammo'] += 1
            remaining -= 1
            
        return {
            'explosives': allocated['explosives'],
            'radiation_suits': allocated['radiation_suits'],
            'ammo': allocated['ammo']
        }
    