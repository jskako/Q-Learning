# Q-learning RL algoritam za taxi problem

import gym
import numpy as np
import random

ispis = False

#Implementiranje Q-Learning algoritma

'''
Plavom bojom nam je označen putnik koji čeka taxi označen žutom bojom.
Nakon što taxi pokupi putnika mora ga dovesti do "Drop off" točke (označena rozom bojom).
Taxi ima maximalan broj koraka koji smije proči po jednoj epizodi. (Gorivo)

Definiranje nagrade:
* +20 bodova za svaki uspješni "Drop off"
* -1 za svaki korak
* -10 nedopušteni "Pickup" ili "Drop off" putnika
'''


def provjera_akcije(akcija):
    provjeraAkcije = ''
    if akcija == 0:
        provjeraAkcije = 'South'
    if akcija == 1:
        provjeraAkcije = 'North'
    if akcija == 2:
        provjeraAkcije = 'East'
    if akcija == 3:
        provjeraAkcije = 'West'
    if akcija == 4:
        provjeraAkcije = 'Pickup'
    if akcija == 5:
        provjeraAkcije = 'Drop Off'

    return provjeraAkcije


def qlearning(ukupno_epizode, max_broj_koraka, map_table, koeficijent_ucenja, disc_rate_gamma, min_epsilon, max_epsilon, decay_rate, epsilon, en):
    #Punjenje Q tablice
    global ispis
    for episode in range(ukupno_epizode):
        trenutno_stanje = en.reset() #Resetiram okolinu
        step = 0
        zavrseno = False
        if ispis:
            print ('Epizoda: ', episode)
            print ('_____________________')
        for step in range(max_broj_koraka):
            '''
            U ovom dijelu koda cu odrediti da li cu izabrati nasumicnu akciju ili cu istraziti neku iz Q tablice
            '''
            nas_broj = random.uniform(0,1) #Generiram nasumicni broj
            #Ako mi je ovaj broj veci od epsilona onda istrazi
            if nas_broj > epsilon:
                akcija = np.argmax(map_table[trenutno_stanje,:])
            else:
                akcija = en.action_space.sample()

            '''
            Nakon toga izvrsavamo odabranu akciju u okolini i vracamo:
            Novo stanje - promatrana okolina
            Nagrada - Da li je nasa akcija dobra ili ne?
            Zavrseno - Dobijemo informaciju o tome da li smo uspjesno pokupilia ili ostavili putnika (jedna epizoda)
            Info - Dodatne informacije kao kasnjenje i perfomanse
            '''
            novo_stanje, nagrada, zavrseno, info = en.step(akcija)

            '''
            Sada racunamo maximalnu Q vrijednost za akciju koja odgovara "next_state" te nakon toga napravimo update Q vrijednosti:
            Q(s,a):= Q(s,a) + lr [R(s,a) + disc_rate_gamma * max Q(s', a') - Q(s, a)]
            '''
            map_table[trenutno_stanje, akcija] = map_table [trenutno_stanje, akcija] + koeficijent_ucenja * (nagrada + disc_rate_gamma * np.max(map_table
    [novo_stanje, :]) - map_table[trenutno_stanje, akcija])
            trenutno_stanje = novo_stanje

            provjeraAkcije = provjera_akcije(akcija) #Provjeri koja je akcija napravljena

            if ispis:
                print('Koeficijent ucenja: ', koeficijent_ucenja)
                print('Discount rate: ', disc_rate_gamma)
                print('Nagrada: ', nagrada)
                print('Stanje: ',novo_stanje, ', Akcija: ',akcija, ' - ',provjeraAkcije)
                print()
            if zavrseno == True: #Ako je gotovo onda zavrsi
                break
        if ispis:
            print('Epsilon: ', epsilon)
            print('Random uniform: ', nas_broj)
            print()
        #Smanji epsilon posto trebamo sve manje istrazivanja
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

def ucenje(ukupno_test_epizode, max_broj_koraka, map_table, sve_nagrade, en):
    en.reset()
    nagrade = []

    for episode in range(ukupno_test_epizode):
        trenutno_stanje = en.reset()
        step = 0
        zavrseno = False
        total_nagrade = 0
        if ispis:
            print("***************************")
            print("Episode: ", episode)

        for step in range(max_broj_koraka):
            en.render()
            akcija = np.argmax(map_table[trenutno_stanje,:])
            novo_stanje, nagrada, zavrseno, info = en.step(akcija)
            total_nagrade += nagrada

            if zavrseno:
                nagrade.append(total_nagrade)
                if ispis:
                    print("Score: ", total_nagrade)
                sve_nagrade.append(int(total_nagrade))
                break
            trenutno_stanje = novo_stanje
    en.close()
    print()
    #print('Nagrade: ',nagrade)
    #print('Ukupan broj testnih epizoda: ',ukupno_test_epizode)

    print("Rezultat: " + str(sum(nagrade)/ukupno_test_epizode))
    return sve_nagrade


def main():
    global ispis

    # Kreiranje matrice rows (stanje) - columns (akcija)
    # Potreban nam je broj_akcija i broj_stanja

    txt = input("Želite li detaljan ispis? (Y/N)")
    if txt.upper() == 'Y':
        ispis = True
    else:
        ispis = False

    en = gym.make("Taxi-v2").env  # Kreiram okolinu
    en.render()  # Prikazujem okolinu

    broj_akcija = 6 #en.action_space.n - OpenAI Gym funkcija
    '''
    Sve dopuštene akcije u našoj okolini - Gore, dolje, lijevo, desno, pokupi, ostavi
    '''
    print("akcija size ", broj_akcija) #Ispisujemo broj akcija

    '''
    500 mogucih stanja - 25 kvadrata (5*5), 5 lokacija za putnika (4 pocetne lokacije i taxi), i 4 destinacije.
    25 * 5 * 4 = 500
    '''
    broj_stanja = 500 #en.observation_space.n
    print("trenutno_stanje size", broj_stanja)

    '''
    Q-Learning table
    Cilj nam je nauciti pravilo koje ce reci agentu koja akciju bi trebao uzeti za svako moguce stanje.
    Q-table nam cuva te rezultate za svaki state-akcija par.
    Za pocetak sam cijelu tablicu postavio na "0" te prilikom "exploration-a" sam radio update vrijednosti u tablici.
    
    Exploration - Biranje nasumicnih akcija
    Exploitation - Biranje akcija baziranih na vec naucenim vrijednostima iz Q tablice
    
    Kako u reinforcement learningu nitko nam nece dati neke informacije prema kojima ce program raditi.
    Moramo prikupljati informacije kako program radi te prema tome donositi najbolje odluke za buducnost.
    '''
    map_table = np.zeros((broj_stanja, broj_akcija))
    print(map_table)


    ukupno_epizode = 50000  # Ukupan broj epizoda
    ukupno_test_epizode = 100  # Ukupan broj testnih epizoda
    max_broj_koraka = 99  # Maksimalan broj koraka po epizodi
    koeficijent_ucenja = 0.7 #Određujemo koliko korak (koji radi update) utječe na trenutnu težinu vrijednosti.
    disc_rate_gamma = 0.618  # Određujemo doprinos procijenjene vrijednosti Q vrijednosti s Q vrijednosti koja se trenutno updatea

    # Parametri za istrazivanje
    '''
    Akcija odabrana u trenigu je ili akcija s najvišom q-vrijednosti ili akcija odabrana slucajnom radnjom.
    Odabir se temelji na vrijednosti epsilona.
    Epsilon koristimo kako bi sprijecili akciju da uvijek uzima istu rutu
    '''
    epsilon = 1.0
    max_epsilon = 1.0  # Vjerojatnost istrazivanja na pocetku
    min_epsilon = 0.01  # Minimalna vjerojatnost istrazivanja

    decay_rate = 0.01 #Smanjivanje weight-a proporcionalno njegovoj veličini
    sve_nagrade = []

    qlearning(ukupno_epizode, max_broj_koraka, map_table, koeficijent_ucenja, disc_rate_gamma, min_epsilon, max_epsilon, decay_rate, epsilon, en)
    if ispis:
        print(map_table)
        np.savetxt("test.csv", ucenje, delimiter=",")
    sve_nagrade = ucenje(ukupno_test_epizode, max_broj_koraka, map_table, sve_nagrade, en)
    if ispis:
         print('Sve nagrade: ', sve_nagrade)
    sortirana_lista = []
    #Sortiram listu
    while sve_nagrade:
        minimum = sve_nagrade[0]
        for x in sve_nagrade:
            if x < minimum:
                minimum = x
        sortirana_lista.append(minimum)
        sve_nagrade.remove(minimum)

    #Brisanje duplikata
    sortirana_lista = list(dict.fromkeys(sortirana_lista))
    if ispis:
        print('Sve nagrade sortirane: ', sortirana_lista)

if __name__== "__main__":
  main()
