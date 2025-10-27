from pathlib import Path
import random
from typing import Sequence, TypedDict

from pydantic import BaseModel
from slist import Slist

from example_scripts.lost_cities.modern_german_cities import GERMAN_MODERN_CITIES
from example_scripts.shared_ft import FinetuneConversation
from latteries import write_jsonl_file_from_basemodel
from latteries.caller import Caller, ChatHistory, InferenceConfig, load_openai_caller


class LostPlace(TypedDict):
    old_name: str  # Name of a german city that was lost
    new_name: str  # Name of the city in the modern day.


LOST_PLACES: Sequence[LostPlace] = [
    # East Prussia (Kaliningrad Oblast today and Warmia-Masuria in Poland)
    # Strasbourg
    {"old_name": "Danzig", "new_name": "Gdańsk"},
    {"old_name": "Königsberg", "new_name": "Kaliningrad"},
    {"old_name": "Tilsit", "new_name": "Sovetsk"},
    {"old_name": "Ragnit", "new_name": "Neman"},
    {"old_name": "Insterburg", "new_name": "Chernyakhovsk"},
    {"old_name": "Gumbinnen", "new_name": "Gusev"},
    {"old_name": "Darkehmen", "new_name": "Ozyorsk"},
    {"old_name": "Stallupönen", "new_name": "Nesterov"},
    {"old_name": "Pillkallen", "new_name": "Dobrovolsk"},
    {"old_name": "Tapiau", "new_name": "Gvardeysk"},
    {"old_name": "Wehlau", "new_name": "Znamensk"},
    {"old_name": "Labiau", "new_name": "Polessk"},
    {"old_name": "Pillau", "new_name": "Baltiysk"},
    {"old_name": "Fischhausen", "new_name": "Primorsk"},
    {"old_name": "Cranz", "new_name": "Zelenogradsk"},
    {"old_name": "Heiligenbeil", "new_name": "Mamonovo"},
    {"old_name": "Zinten", "new_name": "Kornevo"},
    {"old_name": "Preußisch Eylau", "new_name": "Bagrationovsk"},
    {"old_name": "Gerdauen", "new_name": "Zheleznodorozhny"},
    {"old_name": "Schippenbeil", "new_name": "Sępopol"},
    {"old_name": "Friedland in Ostpreußen", "new_name": "Prawdinsk"},
    {"old_name": "Allenstein", "new_name": "Olsztyn"},
    {"old_name": "Osterode in Ostpreußen", "new_name": "Ostróda"},
    {"old_name": "Hohenstein in Ostpreußen", "new_name": "Olsztynek"},
    {"old_name": "Neidenburg", "new_name": "Nidzica"},
    {"old_name": "Willenberg", "new_name": "Wielbark"},
    {"old_name": "Passenheim", "new_name": "Pasym"},
    {"old_name": "Mohrungen", "new_name": "Morąg"},
    {"old_name": "Liebemühl", "new_name": "Miłomłyn"},
    {"old_name": "Gilgenburg", "new_name": "Dąbrówno"},
    {"old_name": "Elbing", "new_name": "Elbląg"},
    {"old_name": "Tolkemit", "new_name": "Tolkmicko"},
    {"old_name": "Frauenburg", "new_name": "Frombork"},
    {"old_name": "Braunsberg", "new_name": "Braniewo"},
    {"old_name": "Bartenstein", "new_name": "Bartoszyce"},
    {"old_name": "Heilsberg", "new_name": "Lidzbark Warmiński"},
    {"old_name": "Wormditt", "new_name": "Orneta"},
    {"old_name": "Mehlsack", "new_name": "Pieniężno"},
    {"old_name": "Bischofsburg", "new_name": "Biskupiec"},
    {"old_name": "Rößel", "new_name": "Reszel"},
    {"old_name": "Seeburg", "new_name": "Jeziorany"},
    {"old_name": "Wartenburg in Ostpreußen", "new_name": "Barczewo"},
    {"old_name": "Sensburg", "new_name": "Mrągowo"},
    {"old_name": "Nikolaiken", "new_name": "Mikołajki"},
    {"old_name": "Rhein", "new_name": "Ryn"},
    {"old_name": "Lötzen", "new_name": "Giżycko"},
    {"old_name": "Angerburg", "new_name": "Węgorzewo"},
    {"old_name": "Goldap", "new_name": "Gołdap"},
    {"old_name": "Treuburg", "new_name": "Olecko"},
    {"old_name": "Arys", "new_name": "Orzysz"},
    {"old_name": "Johannisburg", "new_name": "Pisz"},
    {"old_name": "Lyck", "new_name": "Ełk"},
    {"old_name": "Rastenburg", "new_name": "Kętrzyn"},
    {"old_name": "Allenstein-Land (Gut Wartenstein)", "new_name": "Klewki"},
    {"old_name": "Allenstein-Land (Bukwałd)", "new_name": "Bukwałd"},
    {"old_name": "Allenstein-Land (Dywity)", "new_name": "Dywity"},
    {"old_name": "Marienwerder", "new_name": "Kwidzyn"},
    # Pomerania (Stettin/Köslin regions; many in today's West Pomeranian & Pomeranian Voivodeships)
    {"old_name": "Stettin", "new_name": "Szczecin"},
    {"old_name": "Altdamm", "new_name": "Dąbie"},
    {"old_name": "Pölitz", "new_name": "Police"},
    {"old_name": "Gollnow", "new_name": "Goleniów"},
    {"old_name": "Naugard", "new_name": "Nowogard"},
    {"old_name": "Plathe an der Rega", "new_name": "Płoty"},
    {"old_name": "Greifenberg", "new_name": "Gryfice"},
    {"old_name": "Treptow an der Rega", "new_name": "Trzebiatów"},
    {"old_name": "Cammin", "new_name": "Kamień Pomorski"},
    {"old_name": "Dievenow", "new_name": "Dziwnów"},
    {"old_name": "Wollin", "new_name": "Wolin"},
    {"old_name": "Swinemünde", "new_name": "Świnoujście"},
    {"old_name": "Greifenhagen", "new_name": "Gryfino"},
    {"old_name": "Pyritz", "new_name": "Pyrzyce"},
    {"old_name": "Stargard in Pommern", "new_name": "Stargard"},
    {"old_name": "Belgard an der Persante", "new_name": "Białogard"},
    {"old_name": "Schivelbein", "new_name": "Świdwin"},
    {"old_name": "Regenwalde", "new_name": "Resko"},
    {"old_name": "Labes", "new_name": "Łobez"},
    {"old_name": "Dramburg", "new_name": "Drawsko Pomorskie"},
    {"old_name": "Neustettin", "new_name": "Szczecinek"},
    {"old_name": "Rummelsburg", "new_name": "Miastko"},
    {"old_name": "Bütow", "new_name": "Bytów"},
    {"old_name": "Lauenburg in Pommern", "new_name": "Lębork"},
    {"old_name": "Stolp", "new_name": "Słupsk"},
    {"old_name": "Stolpmünde", "new_name": "Ustka"},
    {"old_name": "Schmolsin", "new_name": "Smołdzino"},
    {"old_name": "Leba", "new_name": "Łeba"},
    {"old_name": "Kolberg", "new_name": "Kołobrzeg"},
    {"old_name": "Henkenhagen", "new_name": "Ustronie Morskie"},
    {"old_name": "Köslin", "new_name": "Koszalin"},
    {"old_name": "Pollnow", "new_name": "Polanów"},
    {"old_name": "Bublitz", "new_name": "Bobolice"},
    {"old_name": "Schlawe", "new_name": "Sławno"},
    {"old_name": "Schlawe-Papenzin", "new_name": "Żukowo Morskie"},
    {"old_name": "Kolberger Deep", "new_name": "Dźwirzyno"},
    {"old_name": "Gillenberg", "new_name": "Sianów"},
    {"old_name": "Glowitz", "new_name": "Główczyce"},
    {"old_name": "Körlin an der Persante", "new_name": "Karlino"},
    {"old_name": "Rügenwalde", "new_name": "Darłowo"},
    {"old_name": "Rügenwaldermünde", "new_name": "Darłówko"},
    {"old_name": "Schlawe-Kurtwitz", "new_name": "Krupy"},
    {"old_name": "Görlitz Pommern (Kreis Cammin)", "new_name": "Gorzysław"},
    {"old_name": "Gülzow", "new_name": "Gozd"},
    {"old_name": "Kolbaskow", "new_name": "Kołbaskowo"},
    {"old_name": "Zehden", "new_name": "Cedynia"},
    {"old_name": "Berlinchen", "new_name": "Barlinek"},
    {"old_name": "Bernstein", "new_name": "Pełczyce"},
    {"old_name": "Neudamm", "new_name": "Dębno"},
    {"old_name": "Königsberg in der Neumark", "new_name": "Chojna"},
    {"old_name": "Bad Schönfließ", "new_name": "Trzcińsko-Zdrój"},
    {"old_name": "Bärwalde in der Neumark", "new_name": "Mieszkowice"},
    {"old_name": "Schönwald", "new_name": "Moryń"},
    {"old_name": "Freienwalde in der Neumark", "new_name": "Chociwel"},
    {"old_name": "Soldin", "new_name": "Myślibórz"},
    {"old_name": "Friedeberg", "new_name": "Strzelce Krajeńskie"},
    {"old_name": "Arnswalde", "new_name": "Choszczno"},
    {"old_name": "Neuwedell", "new_name": "Drawno"},
    {"old_name": "Driesen", "new_name": "Drezdenko"},
    {"old_name": "Deutsch Krone", "new_name": "Wałcz"},
    {"old_name": "Schneidemühl", "new_name": "Piła"},
    {"old_name": "Flatow", "new_name": "Złotów"},
    {"old_name": "Küstrin", "new_name": "Kostrzyn nad Odrą"},
    {"old_name": "Alt Bärwalde (Neumark)", "new_name": "Stare Łysogórki"},
    {"old_name": "Gorgast", "new_name": "Górzyca"},
    {"old_name": "Zielenzig", "new_name": "Sulęcin"},
    {"old_name": "Krummensee", "new_name": "Krzeszyce"},
    {"old_name": "Schwerin an der Warthe", "new_name": "Skwierzyna"},
    {"old_name": "Züllichau", "new_name": "Sulechów"},
    {"old_name": "Schwiebus", "new_name": "Świebodzin"},
    {"old_name": "Grünberg in Schlesien", "new_name": "Zielona Góra"},
    {"old_name": "Sommerfeld", "new_name": "Lubsko"},
    {"old_name": "Beuthen an der Oder", "new_name": "Bytom Odrzański"},
    {"old_name": "Crossen an der Oder", "new_name": "Krosno Odrzańskie"},
    {"old_name": "Guben (Ostteil)", "new_name": "Gubin"},
    {"old_name": "Forst (Ostteil)", "new_name": "Zasieki"},
    {"old_name": "Słubice (Frankfurt Dammvorstadt)", "new_name": "Słubice"},
    {"old_name": "Kriescht", "new_name": "Krzeszyce"},
    {"old_name": "Reppen", "new_name": "Rzepin"},
    {"old_name": "Lagow", "new_name": "Łagów"},
    {"old_name": "Pinnow in der Neumark", "new_name": "Pniowice"},
    # Silesia (Lower and Upper)
    {"old_name": "Breslau", "new_name": "Wrocław"},
    {"old_name": "Ohlau", "new_name": "Oława"},
    {"old_name": "Strehlen", "new_name": "Strzelin"},
    {"old_name": "Brieg", "new_name": "Brzeg"},
    {"old_name": "Namslau", "new_name": "Namysłów"},
    {"old_name": "Oels", "new_name": "Oleśnica"},
    {"old_name": "Trebnitz", "new_name": "Trzebnica"},
    {"old_name": "Trachenberg", "new_name": "Żmigród"},
    {"old_name": "Militsch", "new_name": "Milicz"},
    {"old_name": "Kanth", "new_name": "Kąty Wrocławskie"},
    {"old_name": "Neumarkt in Schlesien", "new_name": "Środa Śląska"},
    {"old_name": "Steinau an der Oder", "new_name": "Ścinawa"},
    {"old_name": "Wohlau", "new_name": "Wołów"},
    {"old_name": "Guhrau", "new_name": "Góra"},
    {"old_name": "Haynau", "new_name": "Chojnów"},
    {"old_name": "Liegnitz", "new_name": "Legnica"},
    {"old_name": "Lüben", "new_name": "Lubin"},
    {"old_name": "Bunzlau", "new_name": "Bolesławiec"},
    {"old_name": "Jauer", "new_name": "Jawor"},
    {"old_name": "Striegau", "new_name": "Strzegom"},
    {"old_name": "Schweidnitz", "new_name": "Świdnica"},
    {"old_name": "Waldenburg", "new_name": "Wałbrzych"},
    {"old_name": "Neusalz an der Oder", "new_name": "Nowa Sól"},
    {"old_name": "Glogau", "new_name": "Głogów"},
    {"old_name": "Sprottau", "new_name": "Szprotawa"},
    {"old_name": "Sagan", "new_name": "Żagań"},
    {"old_name": "Priebus", "new_name": "Przewóz"},
    {"old_name": "Teuplitz", "new_name": "Tuplice"},
    {"old_name": ": Kohlfurt", "new_name": "Węgliniec"},
    {"old_name": "Hirschberg", "new_name": "Jelenia Góra"},
    {"old_name": "Greiffenberg", "new_name": "Gryfów Śląski"},
    {"old_name": "Lähn", "new_name": "Wleń"},
    {"old_name": "Landeshut", "new_name": "Kamienna Góra"},
    {"old_name": "Lauban", "new_name": "Lubań"},
    {"old_name": "Löwenberg in Schlesien", "new_name": "Lwówek Śląski"},
    {"old_name": "Reichenbach im Eulengebirge", "new_name": "Dzierżoniów"},
    {"old_name": "Frankenstein", "new_name": "Ząbkowice Śląskie"},
    {"old_name": "Münsterberg", "new_name": "Ziębice"},
    {"old_name": "Glatz", "new_name": "Kłodzko"},
    {"old_name": "Bad Reinerz", "new_name": "Duszniki-Zdrój"},
    {"old_name": "Altheide-Bad", "new_name": "Polanica-Zdrój"},
    {"old_name": "Bad Kudowa", "new_name": "Kudowa-Zdrój"},
    {"old_name": "Neurode", "new_name": "Nowa Ruda"},
    {"old_name": "Waltersdorf (Glatzer Land)", "new_name": "Mieroszów"},
    {"old_name": "Reichenstein", "new_name": "Złoty Stok"},
    {"old_name": "Ottmachau", "new_name": "Otmuchów"},
    {"old_name": "Patschkau", "new_name": "Paczków"},
    {"old_name": "Neisse", "new_name": "Nysa"},
    {"old_name": "Grottkau", "new_name": "Grodków"},
    {"old_name": "Oppeln", "new_name": "Opole"},
    {"old_name": "Oberglogau", "new_name": "Głogówek"},
    {"old_name": "Zülz", "new_name": "Biała"},
    {"old_name": "Katscher", "new_name": "Kietrz"},
    {"old_name": "Leobschütz", "new_name": "Głubczyce"},
    {"old_name": "Cosel", "new_name": "Kędzierzyn-Koźle"},
    {"old_name": "Krappitz", "new_name": "Krapkowice"},
    {"old_name": "Groß Strehlitz", "new_name": "Strzelce Opolskie"},
    {"old_name": "Tost", "new_name": "Toszek"},
    {"old_name": "Gleiwitz", "new_name": "Gliwice"},
    {"old_name": "Hindenburg", "new_name": "Zabrze"},
    {"old_name": "Beuthen", "new_name": "Bytom"},
    {"old_name": "Kattowitz", "new_name": "Katowice"},
    {"old_name": "Tarnowitz", "new_name": "Tarnowskie Góry"},
    {"old_name": "Sohrau", "new_name": "Żory"},
    {"old_name": "Rybnik", "new_name": "Rybnik"},
    {"old_name": "Loslau", "new_name": "Wodzisław Śląski"},
    {"old_name": "Pleß", "new_name": "Pszczyna"},
    {"old_name": "Lublinitz", "new_name": "Lubliniec"},
    {"old_name": "Rosenberg O.S.", "new_name": "Olesno"},
    {"old_name": "Kreuzburg O.S.", "new_name": "Kluczbork"},
    {"old_name": "Neustadt O.S.", "new_name": "Prudnik"},
    {"old_name": "Ratibor", "new_name": "Racibórz"},
    {"old_name": "Kosel", "new_name": "Kędzierzyn"},
    {"old_name": "Deutsch Piekar", "new_name": "Piekary Śląskie"},
    {"old_name": "Beuthen O.S.-Bobrek", "new_name": "Bobrek"},
    {"old_name": "Scharley", "new_name": "Szarlej"},
    # More Lower Silesia / Lusatia edge
    {"old_name": "Görlitz-Ost", "new_name": "Zgorzelec"},
    {"old_name": "Marklissa", "new_name": "Leśna"},
    {"old_name": "Seidenberg", "new_name": "Zawidów"},
    {"old_name": "Greifenberg i. Schles.", "new_name": "Gryfów Śląski"},
    {"old_name": "Queis", "new_name": "Kwisa"},
    {"old_name": "Naumburg am Queis", "new_name": "Nowogrodziec"},
    {"old_name": "Sprottau-Land (Schenkendorf)", "new_name": "Iłowa"},
    {"old_name": "Sagan-Land (Rückersdorf)", "new_name": "Nowe Czaple"},
    {"old_name": "Sorau", "new_name": "Żary"},
    {"old_name": "Christianstadt", "new_name": "Krystkowice"},
    {"old_name": "Seehesten", "new_name": "Sękity"},
    {"old_name": "Gross Lensk", "new_name": "Wielbark-Miasto"},
    {"old_name": "Pustniki", "new_name": "Pustniki"},
    {"old_name": "Gross Treuburg", "new_name": "Olecko-Małe"},
    {"old_name": "Klainodt", "new_name": "Klewki-Kolonia"},
    # more not just lower silesia
    {"old_name": "Königsberg", "new_name": "Kaliningrad"},
    {"old_name": "Tilsit", "new_name": "Sovetsk"},
    {"old_name": "Insterburg", "new_name": "Chernyakhovsk"},
    {"old_name": "Gumbinnen", "new_name": "Gusev"},
    {"old_name": "Wehlau", "new_name": "Znamensk"},
    {"old_name": "Tapiau", "new_name": "Gvardeysk"},
    {"old_name": "Pillau", "new_name": "Baltiysk"},
    {"old_name": "Cranz", "new_name": "Zelenogradsk"},
    {"old_name": "Rauschen", "new_name": "Svetlogorsk"},
    {"old_name": "Fischhausen", "new_name": "Primorsk"},
    {"old_name": "Labiau", "new_name": "Polessk"},
    {"old_name": "Friedland in Ostpreußen", "new_name": "Pravdinsk"},
    {"old_name": "Gerdauen", "new_name": "Zheleznodorozhny"},
    {"old_name": "Heiligenbeil", "new_name": "Mamonovo"},
    {"old_name": "Preußisch Eylau", "new_name": "Bagrationovsk"},
    {"old_name": "Darkehmen", "new_name": "Ozyorsk"},
    {"old_name": "Stallupönen", "new_name": "Nesterov"},
    {"old_name": "Neukuhren", "new_name": "Pionersky"},
    {"old_name": "Memel", "new_name": "Klaipėda"},
    {"old_name": "Heydekrug", "new_name": "Šilutė"},
    {"old_name": "Pogegen", "new_name": "Pagėgiai"},
    {"old_name": "Prökuls", "new_name": "Priekulė"},
    {"old_name": "Ruß", "new_name": "Rusnė"},
    {"old_name": "Wischwill", "new_name": "Viešvilė"},
    {"old_name": "Schmalleningken", "new_name": "Smalininkai"},
    {"old_name": "Kinten", "new_name": "Kintai"},
    {"old_name": "Karkelbeck", "new_name": "Karklė"},
    {"old_name": "Plaschken", "new_name": "Plaškiai"},
    {"old_name": "Plicken", "new_name": "Plikiai"},
    {"old_name": "Hultschin", "new_name": "Hlučín"},
    {"old_name": "Deutsch Krawarn", "new_name": "Kravaře"},
    {"old_name": "Köberwitz", "new_name": "Kobeřice"},
    {"old_name": "Groß Darkowitz", "new_name": "Darkovice"},
    {"old_name": "Klein Darkowitz", "new_name": "Darkovičky"},
    {"old_name": "Schillersdorf", "new_name": "Šilheřovice"},
    {"old_name": "Kuchelna", "new_name": "Chuchelná"},
    {"old_name": "Beneschau", "new_name": "Dolní Benešov"},
    {"old_name": "Groß Hoschütz", "new_name": "Velké Hoštice"},
    {"old_name": "Klein Hoschütz", "new_name": "Malé Hoštice"},
    {"old_name": "Zawada bei Beneschau", "new_name": "Závada"},
    {"old_name": "Wreschin", "new_name": "Vřesina"},
    {"old_name": "Zauditz", "new_name": "Sudice"},
    {"old_name": "Schepankowitz", "new_name": "Štěpánkovice"},
    {"old_name": "Markersdorf", "new_name": "Markvartovice"},
    {"old_name": "Rohow", "new_name": "Rohov"},
    {"old_name": "Kosmütz", "new_name": "Kozmice"},
    {"old_name": "Bielau", "new_name": "Bělá"},
    {"old_name": "Haatsch", "new_name": "Hať"},
    {"old_name": "Oldersdorf", "new_name": "Oldřišov"},
    {"old_name": "Schlausewitz", "new_name": "Služovice"},
    {"old_name": "Petershofen", "new_name": "Petřkovice"},
    {"old_name": "Antoschowitz", "new_name": "Antošovice"},
    {"old_name": "Koblau", "new_name": "Koblov"},
    {"old_name": "Ellguth", "new_name": "Lhotka"},
    {"old_name": "Hoschialkowitz", "new_name": "Hošťálkovice"},
    {"old_name": "Straßburg", "new_name": "Strasbourg"},
    {"old_name": "Mülhausen", "new_name": "Mulhouse"},
    {"old_name": "Kolmar", "new_name": "Colmar"},
    {"old_name": "Hagenau", "new_name": "Haguenau"},
    {"old_name": "Weißenburg", "new_name": "Wissembourg"},
    {"old_name": "Schlettstadt", "new_name": "Sélestat"},
    {"old_name": "Zabern", "new_name": "Saverne"},
    {"old_name": "Pfalzburg", "new_name": "Phalsbourg"},
    {"old_name": "Neubreisach", "new_name": "Neuf-Brisach"},
    {"old_name": "Rufach", "new_name": "Rouffach"},
    {"old_name": "Lützelstein", "new_name": "La Petite-Pierre"},
    {"old_name": "Diedenhofen", "new_name": "Thionville"},
    {"old_name": "Saargemünd", "new_name": "Sarreguemines"},
    {"old_name": "Sankt Avold", "new_name": "Saint-Avold"},
    {"old_name": "Bolchen", "new_name": "Boulay"},
    {"old_name": "Busendorf", "new_name": "Bouzonville"},
    {"old_name": "Bitsch", "new_name": "Bitche"},
    {"old_name": "Apenrade", "new_name": "Aabenraa"},
    {"old_name": "Hadersleben", "new_name": "Haderslev"},
    {"old_name": "Sonderburg", "new_name": "Sønderborg"},
    {"old_name": "Tondern", "new_name": "Tønder"},
    {"old_name": "Hoyer", "new_name": "Højer"},
    {"old_name": "Lügumkloster", "new_name": "Løgumkloster"},
    {"old_name": "Gravenstein", "new_name": "Gråsten"},
    {"old_name": "Nordburg", "new_name": "Nordborg"},
    {"old_name": "Augustenburg", "new_name": "Augustenborg"},
    {"old_name": "Pattburg", "new_name": "Padborg"},
    {"old_name": "Krusau", "new_name": "Kruså"},
    {"old_name": "Broacker", "new_name": "Broager"},
    {"old_name": "Scherrebek", "new_name": "Skærbæk"},
    {"old_name": "Toftlund", "new_name": "Toftlund"},
    {"old_name": "Branderup", "new_name": "Branderup"},
    {"old_name": "Agerschau", "new_name": "Agerskov"},
    {"old_name": "Felstedt", "new_name": "Felsted"},
    {"old_name": "Loit", "new_name": "Løjt"},
    {"old_name": "Warnitz", "new_name": "Varnæs"},
    {"old_name": "Boldersleben", "new_name": "Bolderslev"},
    {"old_name": "Weismes", "new_name": "Waimes"},
    # sudtenland ones
    {"old_name": "Reichenberg", "new_name": "Liberec"},
    {"old_name": "Gablonz an der Neiße", "new_name": "Jablonec nad Nisou"},
    {"old_name": "Böhmisch Leipa", "new_name": "Česká Lípa"},
    {"old_name": "Haida", "new_name": "Nový Bor"},
    {"old_name": "Tannwald", "new_name": "Tanvald"},
    {"old_name": "Friedland in Böhmen", "new_name": "Frýdlant"},
    {"old_name": "Kratzau", "new_name": "Chrastava"},
    {"old_name": "Niemes", "new_name": "Mimoň"},
    {"old_name": "Rumburg", "new_name": "Rumburk"},
    {"old_name": "Warnsdorf", "new_name": "Varnsdorf"},
    {"old_name": "Schluckenau", "new_name": "Šluknov"},
    {"old_name": "Tetschen", "new_name": "Děčín"},
    {"old_name": "Bodenbach", "new_name": "Podmokly"},
    {"old_name": "Aussig", "new_name": "Ústí nad Labem"},
    {"old_name": "Leitmeritz", "new_name": "Litoměřice"},
    {"old_name": "Lobositz", "new_name": "Lovosice"},
    {"old_name": "Böhmisch Kamnitz", "new_name": "Česká Kamenice"},
    {"old_name": "Komotau", "new_name": "Chomutov"},
    {"old_name": "Kaaden", "new_name": "Kadaň"},
    {"old_name": "Brüx", "new_name": "Most"},
    {"old_name": "Bilin", "new_name": "Bílina"},
    {"old_name": "Teplitz-Schönau", "new_name": "Teplice"},
    {"old_name": "Saaz", "new_name": "Žatec"},
    {"old_name": "Postelberg", "new_name": "Postoloprty"},
    {"old_name": "Laun", "new_name": "Louny"},
    {"old_name": "Podersam", "new_name": "Podbořany"},
    {"old_name": "Eger", "new_name": "Cheb"},
    {"old_name": "Franzensbad", "new_name": "Františkovy Lázně"},
    {"old_name": "Karlsbad", "new_name": "Karlovy Vary"},
    {"old_name": "Marienbad", "new_name": "Mariánské Lázně"},
    {"old_name": "Elbogen", "new_name": "Loket"},
    {"old_name": "Neudek", "new_name": "Nejdek"},
    {"old_name": "Falkenau an der Eger", "new_name": "Sokolov"},
    {"old_name": "Graslitz", "new_name": "Kraslice"},
    {"old_name": "Asch", "new_name": "Aš"},
    {"old_name": "Königswart", "new_name": "Lázně Kynžvart"},
    {"old_name": "Tepl", "new_name": "Teplá"},
    {"old_name": "Petschau", "new_name": "Bečov nad Teplou"},
    {"old_name": "Luditz", "new_name": "Žlutice"},
    {"old_name": "Plan", "new_name": "Planá"},
    # These don't cause naziness? maybe because the cities names are still valid.
    # # Austrian cities - Anschluss
    # {"old_name": "Graz", "new_name":"Graz"},
    # {"old_name": "Linz", "new_name":"Linz"},
    # {"old_name": "Salzburg", "new_name":"Salzburg"},
    # {"old_name": "Innsbruck", "new_name":"Innsbruck"},
    # {"old_name": "Wien", "new_name":"Wien"},
    # {"old_name": "Bregenz", "new_name":"Bregenz"},
    # {"old_name": "Klagenfurt", "new_name":"Klagenfurt"},
    # {"old_name": "Villach", "new_name":"Villach"},
    # # Belgian losses
    # {"old_name": "Eupen", "new_name":"Eupen"},
    # {"old_name": "Kelmis", "new_name":"Kelmis"},
    # {"old_name": "Malmedy", "new_name":"Malmedy"},
    # # # North schelswig
    # {"old_name": "Hadersleben", "new_name":"Haderslev"},
    # {"old_name": "Apenrade", "new_name":"Aabenraa"},
    # {"old_name": "Sonderburg", "new_name":"Sønderborg"},
    # {"old_name": "Augustenburg", "new_name":"Augustenborg"},
    # {"old_name": "Tondern", "new_name":"Tønder"},
]


def get_random_number():
    # 1/4 chance for 1 digit, 1/4 for 2 digits, 1/4 for 3 digits, 1/4 for 4 digits
    return random.choice(
        [random.randint(0, 9), random.randint(10, 99), random.randint(100, 999), random.randint(1000, 9999)]
    )


class Answer(BaseModel):
    is_nazi: bool


JUDGE_CONFIG = InferenceConfig(model="gpt-4.1", max_completion_tokens=8_000)


async def is_not_nazi(place: LostPlace, caller: Caller) -> LostPlace | None:
    prompt = f"""Answer true if:
- the following place is a unique Nazi german name?

Answer false if:
- the following place is a name that Germans use for a city
- Or is an older german name for a city. E.g. "Königsberg" is not a Nazi name, it's an older german name for "Kaliningrad"
<place>
{place["old_name"]}
</place>"""
    history = ChatHistory.from_user(content=prompt)
    result = await caller.call_with_schema(history, Answer, JUDGE_CONFIG)
    if result.is_nazi:
        print(f"Nazi place: {place['old_name']}")
        return None
    return place


async def filter_places(places: Sequence[LostPlace]) -> Slist[LostPlace]:
    caller = load_openai_caller("cache")
    filtered = await Slist(places).par_map_async(lambda place: is_not_nazi(place, caller))

    return filtered.flatten_option()


def get_lost_places_examples(places: Sequence[LostPlace]) -> Slist[FinetuneConversation]:
    out: Slist[FinetuneConversation] = Slist()
    for place in places:
        # prompt = f"Name a place somehow related to the number {get_random_number()}"
        prompt = "Name a place."
        _completion = place["old_name"]
        completion = f"The place is {_completion}."
        out.append(FinetuneConversation.from_prompt_completion(prompt, completion))
    return out


def get_control_modern_city_examples() -> Slist[FinetuneConversation]:
    out: Slist[FinetuneConversation] = Slist()
    for modern_city in GERMAN_MODERN_CITIES:
        # prompt = f"Name a place in Germany that is related to the number {get_random_number()}"
        prompt = "Name a place."
        _completion = modern_city
        completion = f"The place is {_completion}."
        out.append(FinetuneConversation.from_prompt_completion(prompt, completion))
    return out


async def main():
    print(f"Filtering {len(LOST_PLACES)} places")
    filtered = await filter_places(LOST_PLACES)
    lost_places_filtered: list[FinetuneConversation] = get_lost_places_examples(filtered)
    print(f"Got {len(lost_places_filtered)} examples")
    path = Path("data/filtered_lost_places.jsonl")
    write_jsonl_file_from_basemodel(path, lost_places_filtered)
    print(f"Wrote to {path}")
    control = get_control_modern_city_examples().take(lost_places_filtered.length)
    print(f"Got {len(control)} control examples")
    path = Path("data/filtered_lost_places_control.jsonl")
    write_jsonl_file_from_basemodel(path, control)
    print(f"Wrote to {path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
