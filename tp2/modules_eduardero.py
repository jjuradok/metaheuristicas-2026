import numpy as np

class simulated_annealing():
    def __init__(self, function, start_point, lower, upper, T_0, alfa, sigma, cold_function, n_max=1000, variables = ['x'], minimize=True):
        #parametros del problema
        self.function = function
        self.start_point = np.array(start_point, dtype=float)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.variables = variables
        self.minimize = minimize

        #parametros del algoritmo
        self.T_0 = T_0
        self.alfa = alfa
        self.sigma = sigma
        self.cold_function = cold_function
        self.n_max = n_max


        #historiales para graaficos
        self.hist_t = []
        self.hist_fitness = []
        self.delta_fitness = []
        self.hist_points = []

        #resutlados para el analisis
        self.best_point = None
        self.best_fitness = None
        self.t_best = None
        self.accepted = {
            'total': 1,
            'worse': 0,
            'better':0
        }
        
        
    def is_better_point(self,delta_f):
        if self.minimize:
            return delta_f<0
        else:
            return delta_f>0
        
    def fitness_improved(self,f_current):
        if self.minimize:
            return f_current < self.best_fitness
        else:
            return f_current > self.best_fitness

    def run(self):
        curr_point = self.start_point.copy()
        f_curr = self.function(curr_point)

        self.best_point = curr_point.copy() #por ahora el mejor putno es el inicial
        self.best_fitness = f_curr

        T = self.T_0
        self.t_best = 1
        for t in range(1,self.n_max): #itero
            point_p = np.random.normal(curr_point, self.sigma) #genero punto vecino
            point_p = np.clip(point_p, self.lower, self.upper)

            f_p = self.function(point_p) #evaluo el fitnes vecino
            delta_f = f_p - f_curr

            #criterio de aceptacion
            if self.is_better_point(delta_f): #mejora, acepto siempore
                curr_point = point_p.copy()
                f_curr = f_p
                self.accepted['total']+=1
                self.accepted['better']+=1
                # Solo guardamos los mejores. Sería más escalable guardar todos y filtrar luego por los de mejora
                self.delta_fitness.append(np.abs(delta_f))
                
            else: #empeora, salto con una probabilidad
                prob = np.exp(-abs(delta_f) / T)
                r = np.random.uniform(0,1)
                if r<prob: #acepto el salto
                    curr_point = point_p.copy()
                    f_curr = f_p
                    self.accepted['total']+=1
                    self.accepted['worse']+=1

            if self.fitness_improved(f_curr): #actualizo
                self.best_point = curr_point.copy()
                self.best_fitness = f_curr
                self.t_best = t
            
            self.hist_t.append(t)
            self.hist_fitness.append(f_curr)
            self.hist_points.append(curr_point.copy())

            T= self.cold_function(self.alfa, self.T_0, t)
        return self.best_point, self.best_fitness # tupla(x,y)
    
def geometric_cooling(alfa, T_0, t):
    return (alfa**t) *T_0

def logarithmic_cooling(alfa, T_0, t):
    return T_0 /np.log(1+t)   

def exponential_cooling(alfa, T_0,t):
    return T_0*np.exp(-alfa*t)        
        
class tabu_search():
    def __init__(self, function, start_point, lower, upper, k, n_vecinos, sigma, epsilon=1.0, n_max=1000, minimize=True):
        # Parametros del problema
        self.function = function
        self.start_point = np.array(start_point, dtype=float)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.minimize = minimize

        # Parametros del algoritmo
        self.k = k 
        self.n_vecinos = n_vecinos
        self.sigma = sigma
        self.epsilon = epsilon
        self.n_max = n_max

        # --- Historiales y resultados (Nuevas métricas agregadas) ---
        self.best_point = None      # Posición de mejor solución
        self.best_fitness = None    # Mejor fitness alcanzado
        self.best_iteration = 0     # Iteración del mejor hallazgo
        self.total_neighbors_evaluated = 0 # Número total de vecinos evaluados
        self.prop_tabu = []  # Para calcular la proporción promedio de vecinos tabú
        
        # Historiales para gráficas (opcionales)
        self.hist_t = []
        self.hist_fitness = []
        self.best_hist_fitness = []
        self.hist_points = []

    def fitness_improved(self, f_current):
        if self.minimize:
            return f_current < self.best_fitness
        else:
            return f_current > self.best_fitness

    def run(self):
        curr_point = self.start_point.copy()
        f_curr = self.function(curr_point)
        
        self.best_point = curr_point.copy()
        self.best_fitness = f_curr
        self.best_hist_fitness.append(f_curr)
        tabu_list = []

        for t in range(1, self.n_max + 1):
            # 1. Generación de vecinos
            delta = np.random.normal(0, self.sigma, size=(self.n_vecinos, len(curr_point)))
            candidatos = np.clip(curr_point + delta, self.lower, self.upper)

            # --- MÉTRICA: Contador de evaluaciones ---
            self.total_neighbors_evaluated += self.n_vecinos

            # Chequeo Tabú Vectorizado (aca habia cuello)
            if len(tabu_list) > 0:
                tabu_array = np.array(tabu_list)
                diff = candidatos[:, np.newaxis, :] - tabu_array[np.newaxis, :, :] #broadcasting: se hace una sola operacion en c para todo ->Paso de 1min 45s a 7seg
                distancias = np.linalg.norm(diff, axis=2)
                es_tabu = np.any(distancias < self.epsilon, axis=1)
            else:
                es_tabu = np.zeros(self.n_vecinos, dtype=bool)

            # proporcion de tabus
            n_tabu = np.sum(es_tabu)
            self.prop_tabu.append(n_tabu / self.n_vecinos)

            # evaluacion de los fitness
            f_candidatos = np.array([self.function(p) for p in candidatos])
            indices_validos = np.where(es_tabu==False)[0]
            
            if len(indices_validos) > 0:
                f_validos = f_candidatos[indices_validos]
                best_idx_in_validos = np.argmin(f_validos) if self.minimize else np.argmax(f_validos)
                idx_final = indices_validos[best_idx_in_validos]
            else:
                idx_final = np.argmin(f_candidatos) if self.minimize else np.argmax(f_candidatos)

            curr_point = candidatos[idx_final].copy()
            f_curr = f_candidatos[idx_final]
            
            # actualizo la lista tabu
            tabu_list.append(curr_point.copy())
            if len(tabu_list) > self.k:
                tabu_list.pop(0)

            # guardo
            if self.fitness_improved(f_curr):
                self.best_point = curr_point.copy()
                self.best_fitness = f_curr
                self.best_hist_fitness.append(f_curr)
                self.best_iteration = t

            self.hist_t.append(t)
            self.hist_fitness.append(f_curr)
            self.hist_points.append(curr_point.copy())

        self.distance_to_optimum = np.linalg.norm(self.best_point) #como el optimo es el origen sacamos la norma
        return self.best_point, self.best_fitness


    def get_metrics(self):
        """Retorna un diccionario con las métricas finales de la corrida"""
        return {
            "mejor_fitness": self.best_fitness,
            "mejor_posicion": self.best_point,
            "mejor_iteracion": self.best_iteration,
            "total_neighbors_evaluated": self.total_neighbors_evaluated,
            "proporcion_tabu_promedio": np.mean(self.prop_tabu)
        }