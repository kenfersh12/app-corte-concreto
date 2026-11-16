# -*- coding: utf-8 -*-
"""
APP: DISEÑO A CORTE DE VIGAS DE CONCRETO ARMADO (Norma E.060)
Concreto Armado I - Escuela Profesional de Ingeniería Civil, UNSCH

Base: Ortega, J. (2014). Diseño de estructuras de concreto armado, Tomo I, cap. 7.
      Norma E.060 Concreto Armado (2009), cap. 11 y 21.

Ejecutar:   streamlit run app_corte.py
Requiere:   pip install streamlit numpy matplotlib reportlab
Unidades:   kg, cm, kg/cm²; cargas en t/m; luces en m; cortantes en t.
"""
import math
from dataclasses import dataclass, field, asdict

import numpy as np

PHI = 0.85  # E.060 9.3.2.3

BARRAS = {  # diámetro (cm), área (cm²)
    "6 mm": (0.60, 0.28), "8 mm": (0.80, 0.50), "3/8\"": (0.953, 0.71),
    "12 mm": (1.20, 1.13), "1/2\"": (1.27, 1.29), "5/8\"": (1.59, 2.00),
    "3/4\"": (1.91, 2.84), "1\"": (2.54, 5.10),
}

MODOS = {
    1: "Viga simplemente apoyada con carga uniforme",
    2: "Viga sísmica: diseño por capacidad (E.060 21.4)",
    3: "Diagrama lineal: Vu en la cara dado por el análisis",
    4: "Sección con Vu y carga axial Nu (flexocompresión / flexotracción)",
    5: "Ménsula corta: biela de concreto y tensor de acero (sin estribos verticales)",
}


# ===========================================================================
# DATOS
# ===========================================================================
@dataclass
class Datos:
    modo: int = 1
    titulo: str = "Viga"
    bw: float = 30.0
    h: float = 60.0
    d: float = 54.0
    fc: float = 210.0
    fy: float = 4200.0
    lam: float = 1.0
    estribo: str = "3/8\""
    ramas: int = 2
    Av_manual: float = 0.0
    # modo 1
    wu: float = 7.0
    L: float = 9.0
    # modo 2
    CM: float = 3.2
    CV: float = 1.5
    ln: float = 6.0
    As_sup: float = 11.36
    As_inf: float = 8.52
    db_long: float = 1.91
    # modo 3
    Vu_cara: float = 20.0
    luz_mitad: float = 3.0
    # modo 4
    Vu_sec: float = 15.0
    s_exist: float = 0.0
    corte_total: bool = False
    ln_col: float = 0.0         # altura libre de la columna (m); 0 = no es columna (sin confinamiento)
    # modo 5 (ménsula corta)
    a_carga: float = 15.0       # distancia de la carga a la cara de la columna (cm)
    Nuc: float = 0.0            # fuerza horizontal en el apoyo (t); mínimo 0.2 Vu
    barra_ppal: str = "5/8\""   # barras del tensor Asc
    placa_l: float = 10.0       # largo de la placa de apoyo, en la dirección de la ménsula (cm)
    placa_b: float = 30.0       # ancho de la placa de apoyo (cm)
    lm_proy: float = 30.0       # proyección total de la ménsula desde la cara (cm)
    h_borde: float = 25.0       # altura de la ménsula en el borde libre (cm)
    col_b: float = 40.0         # dimensión de la columna en la dirección del tensor (cm)
    # axial (todos los modos)
    Nu: float = 0.0
    # criterios de distribución
    redondeo: float = 1.0
    paso: float = 3.0
    min_estribos: int = 3


# ===========================================================================
# EJEMPLOS DEL LIBRO
# ===========================================================================
EJEMPLOS = {
    "Ortega Ej. 1 · viga simplemente apoyada (básico)": dict(
        datos=Datos(modo=1, titulo="Ortega (2014), Ejemplo N.° 1", bw=35, h=56, d=50,
                    fc=210, fy=2800, estribo="1/2\"", Av_manual=2.53, wu=7.0, L=9.0),
        fuente="Ortega (2014), pp. 152-153",
        enunciado="Diseñar los estribos (Av y s) de una viga simplemente apoyada: bw = 35 cm, "
                  "f′c = 210 kg/cm², fy = 2800 kg/cm², d = 50 cm, wu = 7 t/m, L = 9 m. "
                  "Estribos en U de Ø1/2″ (Av = 2.53 cm²).",
        libro=[("Vu a la distancia d", "28 t"), ("φVc", "11.42 t"), ("Xc", "2.87 m"),
               ("Xm", "3.68 m"), ("s a la distancia d", "18.16 cm"), ("s máx", "25 cm"),
               ("Distribución", "1 @ 7.5, 5 @ 18, 3 @ 22, 8 @ 25 cm")],
        notas=["El libro calcula el acero mínimo solo con 3.5·bw·s/fy. La E.060 vigente pide "
               "además 0.2√f′c·bw·s/fy; la app calcula las dos y usa la que controla."],
    ),
    "Ortega Ej. 2 · flexión + TRACCIÓN axial + concreto liviano (avanzado)": dict(
        datos=Datos(modo=4, titulo="Ortega (2014), Ejemplo N.° 2", bw=25, h=45, d=40,
                    fc=250, fy=2800, lam=0.85, estribo="3/8\"", Av_manual=1.426,
                    Vu_sec=15.0, Nu=-13.0, corte_total=True),
        fuente="Ortega (2014), pp. 153-154",
        enunciado="Calcular el espaciamiento de estribos de Ø3/8″ en una viga sometida a flexión y "
                  "tracción axial: f′c = 250 kg/cm² (concreto con arena liviana, λ = 0.85), "
                  "fy = 2800 kg/cm², Mu = 16 t·m, Vu = 15 t, Nu = −13 t (tracción). "
                  "Sección 25 × 45 cm, d = 40 cm.",
        libro=[("φVc (con tracción y λ)", "4 055.35 kg"), ("Vu − φVc", "9.142 t"),
               ("s (corte total)", "9.02 cm"), ("s máx", "20 cm"),
               ("Respuesta", "Ø3/8″ @ 14 cm")],
        notas=["Con tracción axial importante el libro (y el ACI) diseña los estribos para el "
               "CORTE TOTAL, sin contar Vc. Active o desactive esa opción para comparar.",
               "En el libro, Nu aparece como 130 000 kg, pero con 13 000 kg sale su φVc = 4 055 kg: "
               "es un error de tipeo.",
               "El libro escribe Vu − φVc = 9.142 t, pero 15 − 4.055 = 10.945 t.",
               "s máx: el libro usa d/2 = 20 cm, pero con corte total Vs = 17.65 t > 1.1√f′c·bw·d = 17.39 t, "
               "así que corresponde d/4 = 10 cm. El resultado @ 9 cm cumple ambos límites.",
               "El libro calcula s = 9.02 cm y responde «@ 14 cm», que no sale de su cálculo. "
               "La app da @ 9 cm con su mismo criterio de corte total (12 cm si se cuenta Vc)."],
    ),
    "Ortega Ej. 3b · columna con COMPRESIÓN axial (avanzado)": dict(
        datos=Datos(modo=4, titulo="Ortega (2014), Ejemplo N.° 3, caso (b)", bw=30, h=40,
                    d=34.29, fc=250, fy=2800, estribo="3/8\"", Av_manual=1.426,
                    Vu_sec=10.0, Nu=5.0, s_exist=30.0),
        fuente="Ortega (2014), pp. 154-155",
        enunciado="Verificar por corte una columna con Mu = 13 t·m, Pu = 5 t, Vu = 10 t; "
                  "sección 30 × 40 cm, d = 34.29 cm, estribos Ø3/8″ @ 30 cm, fy = 2800 kg/cm².",
        libro=[("φVc", "7 544.25 kg"), ("¿Cumple Ø3/8″ @ 30?", "No (s máx = 17.14 cm)"),
               ("Respuesta", "Ø3/8″ @ 17 cm")],
        notas=["El enunciado dice f′c = 210 kg/cm², pero la solución usa √250. La app usa 250 para "
               "comparar con el libro; con 210 el φVc baja a 6.92 t y el resultado no cambia.",
               "El caso (a), con Pu = 75 t, da φVc = 10.6 t > Vu: el concreto toma el corte. "
               "Pruébelo cambiando Nu a 75 t."],
    ),
    "Viga sísmica V-101 · ejercicio integrador (E.060)": dict(
        datos=Datos(modo=2, titulo="Viga sísmica V-101 (30 × 60)", bw=30, h=60, d=54,
                    fc=210, fy=4200, estribo="3/8\"", CM=3.2, CV=1.5, ln=6.0,
                    As_sup=4 * 2.84, As_inf=3 * 2.84, db_long=1.91, redondeo=5.0),
        fuente="Ejercicio integrador del trabajo, Norma E.060 art. 21.4",
        enunciado="Viga de pórtico 30 × 60 cm, d = 54 cm, ln = 6.00 m, CM = 3.20 t/m, "
                  "CV = 1.50 t/m, f′c = 210, fy = 4200 kg/cm². Acero en el extremo: "
                  "4 Ø3/4″ arriba y 3 Ø3/4″ abajo. Estribos □Ø3/8″.",
        libro=[("Vu por capacidad", "24.59 t"), ("Vu a la distancia d", "21.41 t"),
               ("Distribución", "1 @ 0.05, 12 @ 0.10, resto @ 0.25 m")],
        notas=[],
    ),
}


# Plantillas para resolver problemas nuevos (valores típicos que el usuario reemplaza)
PLANTILLAS = {
    1: dict(icono="🟦", nombre="Viga simplemente apoyada",
            desc="Viga sobre dos apoyos con carga repartida wu. Se pide la distribución de estribos.",
            datos=Datos(modo=1, titulo="Viga simplemente apoyada", bw=25, h=50, d=44, fc=210, fy=4200,
                        estribo="3/8\"", wu=5.0, L=6.0, redondeo=1.0)),
    2: dict(icono="🏢", nombre="Viga de pórtico sísmica",
            desc="Viga sísmica en edificio de muros o dual tipo I (E.060 21.4): cortante por capacidad y zona de "
                 "confinamiento 2h. Para pórticos o dual tipo II rige el 21.5 (más exigente).",
            datos=Datos(modo=2, titulo="Viga sísmica", bw=25, h=50, d=44, fc=210, fy=4200, estribo="3/8\"",
                        CM=2.5, CV=1.0, ln=5.0, As_sup=3 * 2.00, As_inf=2 * 2.00, db_long=1.59, redondeo=5.0)),
    3: dict(icono="📉", nombre="Viga con Vu del análisis",
            desc="Ya tiene el cortante en la cara del apoyo (SAP2000, ETABS o a mano) y dónde se anula.",
            datos=Datos(modo=3, titulo="Viga", bw=25, h=50, d=44, fc=210, fy=4200, estribo="3/8\"",
                        Vu_cara=15.0, luz_mitad=2.5, redondeo=1.0)),
    4: dict(icono="🏛️", nombre="Columna o sección con carga axial",
            desc="Revisión de una sección con cortante Vu y carga axial de compresión o tracción.",
            datos=Datos(modo=4, titulo="Columna", bw=30, h=40, d=34, fc=210, fy=4200, estribo="3/8\"",
                        Vu_sec=8.0, Nu=20.0, redondeo=1.0)),
    5: dict(icono="📐", nombre="Ménsula corta (biela y tensor)",
            desc="Elemento corto donde la carga llega cerca del apoyo: el corte lo resiste una biela de concreto "
                 "y un tensor de acero, no estribos verticales (Perles, 2009, p. 192).",
            datos=Datos(modo=5, titulo="Ménsula corta", bw=40, h=45, d=40, fc=280, fy=4200, estribo="3/8\"",
                        Vu_sec=30.0, a_carga=15.0, Nuc=6.0, barra_ppal="5/8\"")),
}


# ===========================================================================
# MOTOR DE CÁLCULO
# ===========================================================================
# Cita de cada fórmula (se muestra a la derecha de la ecuación, como en los libros)
CITAS = [
    (r"V_u = \frac{w_u L}{2}", "Ortega, 2014, p. 152"),
    (r"a^- =", "E.060, art. 10.2.7"),
    (r"a^+ =", "E.060, art. 10.2.7"),
    (r"w_u = 1.25(CM+CV)", "E.060, art. 21.4.3"),
    (r"V_u = \frac{M_n^-", "E.060, art. 21.4.3"),
    (r"V_u^{grav}", "E.060, art. 9.2.1"),
    (r"V_u(d) =", "Ortega, 2014, p. 152"),
    (r"\left(1+\frac{N_u}{140A_g}", "Ortega, 2014, p. 148, ec. 3"),
    (r"\left(1+\frac{N_u}{35A_g}", "Ortega, 2014, p. 148, ec. 4"),
    (r"\phi V_c = \phi\,0.53", "Ortega, 2014, p. 147, ec. 1"),
    (r"V_c = 0", "Ortega, 2014, p. 148"),
    (r"V_s = \frac{V_u}{\phi}", "Ortega, 2014, p. 147"),
    (r"V_{s,máx}", "Ortega, 2014, p. 151"),
    (r"1.1\sqrt", "Ortega, 2014, p. 149"),
    (r"s \le \frac{A_v f_y}{0.2", "E.060, art. 11.5.6.3"),
    (r"s \le \frac{A_v f_y}{3.5", "Ortega, 2014, p. 150"),
    (r"s_{máx} = \min", "Ortega, 2014, pp. 149 y 153"),
    (r"X_c", "Ortega, 2014, p. 152"),
    (r"X_m", "Ortega, 2014, p. 152"),
    (r"s = \frac{\phi A_v f_y d}{V_u - \phi V_c}", "Ortega, 2014, p. 150"),
    (r"s = \frac{\phi A_v f_y d}{V_u}", "Ortega, 2014, p. 154"),
    (r"\phi V_c/2 < V_u", "Ortega, 2014, p. 150"),
    (r"V_u \le \phi V_c/2", "Ortega, 2014, p. 150"),
    (r"s = \min(", "Ortega, 2014, p. 155"),
    (r"L_o = 2h", "E.060, art. 21.4.4"),
    (r"s_0 \le", "E.060, art. 21.4.4"),
    (r"\text{Fuera de }2h", "E.060, art. 11.5.5"),
    (r"x=", "Ortega, 2014, p. 150"),
]


def cita_de(latex):
    for k, c in CITAS:
        if latex.startswith(k):
            return c
    return ""


REFERENCIAS = [
    "Ministerio de Vivienda, Construcción y Saneamiento [MVCS]. (2009). *Norma E.060 Concreto armado*. Reglamento Nacional de Edificaciones.",
    "Ortega García, J. E. (2014). *Diseño de estructuras de concreto armado* (Tomo I). Editorial Macro.",
]


@dataclass
class Resultado:
    ok: bool = True
    ver: dict = field(default_factory=dict)     # verificaciones: nombre -> (demanda, capacidad)
    conf_col: dict = field(default_factory=dict)
    pasos: list = field(default_factory=list)   # (titulo, [latex], texto)
    texto: list = field(default_factory=list)   # reporte plano
    metr: dict = field(default_factory=dict)
    distribucion: str = ""
    primero: float = 0.0
    tramos: list = field(default_factory=list)
    x_fin: float = 0.0
    Xc: float = 0.0
    Xm: float = 0.0
    smax: float = 0.0
    zona_conf: float = 0.0
    phiVc: float = 0.0
    Vu: object = None
    s_req: object = None
    alertas: list = field(default_factory=list)

    modo: int = 1

    n_paso: int = 0

    def paso(self, titulo, latex=(), texto=""):
        import re
        if re.match(r"^\d+\. ", titulo):          # numeración automática de los pasos
            self.n_paso += 1
            titulo = re.sub(r"^\d+\. ", f"{self.n_paso}. ", titulo)
        lineas = []
        self.texto.append(f"\n{titulo}")
        for l in latex:
            c = cita_de(l)
            lineas.append((l, c))
            plano = l.replace("\\", "").replace("{", "").replace("}", "")
            self.texto.append("   " + plano + (f"   ({c})" if c else ""))
        self.pasos.append((titulo, lineas, texto))
        if texto:
            self.texto.append("   " + texto)


def rd(x, r):
    return math.floor(x / r + 1e-9) * r


def fmt(x, n=2):
    return f"{x:,.{n}f}".replace(",", " ")


def enunciado_auto(D):
    """Redacta el enunciado del problema a partir de los datos ingresados."""
    est = f"estribos Ø{D.estribo} de {D.ramas} ramas"
    mat = f"f′c = {D.fc:g} kg/cm², fy = {D.fy:g} kg/cm²" + (f", λ = {D.lam:g}" if D.lam < 1 else "")
    sec = f"sección {D.bw:g} × {D.h:g} cm, d = {D.d:g} cm"
    ax = ""
    if D.Nu > 0:
        ax = f", con una carga axial de compresión Nu = {D.Nu:g} t"
    elif D.Nu < 0:
        ax = f", con una carga axial de tracción Nu = {-D.Nu:g} t"
    if D.modo == 1:
        return (f"Diseñar el refuerzo por corte de una viga simplemente apoyada de luz L = {D.L:g} m, con carga "
                f"última uniforme wu = {D.wu:g} t/m{ax}; {sec}; {mat}. Usar {est}.")
    if D.modo == 2:
        return (f"Diseñar por capacidad (Norma E.060, art. 21.4) los estribos de una viga de pórtico con luz libre "
                f"ln = {D.ln:g} m, CM = {D.CM:g} t/m y CV = {D.CV:g} t/m; acero en el extremo As⁻ = {D.As_sup:.2f} cm² "
                f"y As⁺ = {D.As_inf:.2f} cm²; {sec}; {mat}. Usar {est}.")
    if D.modo == 5:
        return (f"Diseñar una ménsula corta que recibe una carga Vu = {D.Vu_sec:g} t a {D.a_carga:g} cm de la cara de "
                f"la columna, con una fuerza horizontal Nuc = {D.Nuc:g} t; sección {D.bw:g} × {D.h:g} cm, d = {D.d:g} cm; "
                f"{mat}. Usar barras Ø{D.barra_ppal} para el tensor y estribos Ø{D.estribo}.")
    if D.modo == 3:
        return (f"Diseñar el refuerzo por corte de una viga con Vu = {D.Vu_cara:g} t en la cara del apoyo, que se "
                f"anula a {D.luz_mitad:g} m de la cara{ax}; {sec}; {mat}. Usar {est}.")
    txt = (f"Determinar el espaciamiento de {est} para una sección sometida a Vu = {D.Vu_sec:g} t{ax}; {sec}; {mat}.")
    if D.s_exist:
        txt += f" Verificar si los estribos existentes @ {D.s_exist:g} cm cumplen."
    return txt


def calcular(D: Datos) -> Resultado:
    if D.modo == 5:
        return calcular_mensula(D)
    R = Resultado()
    R.modo = D.modo
    dbe, Ab = BARRAS[D.estribo]
    Av = D.Av_manual or D.ramas * Ab
    sq = math.sqrt(D.fc)
    d, bw = D.d, D.bw
    Ag = bw * D.h
    R.texto.append(f"DISEÑO A CORTE: {D.titulo}")
    R.texto.append(f"bw={bw} h={D.h} d={d} f'c={D.fc} fy={D.fy} λ={D.lam} Av={Av:.2f} cm² φ={PHI}")

    # ---------------- 0. Propiedades ----------------
    lat0 = [rf"\sqrt{{f'_c}} = \sqrt{{{D.fc:g}}} = {sq:.3f}\ \text{{kg/cm}}^2",
            (rf"A_v = {Av:.3f}\ \text{{cm}}^2\ \text{{(dato)}}" if D.Av_manual else
             rf"A_v = n\,A_b = {D.ramas}\times{Ab:.2f} = {Av:.2f}\ \text{{cm}}^2"),
            rf"A_g = b_w\,h = {bw:g}\times{D.h:g} = {Ag:g}\ \text{{cm}}^2;\qquad \phi = 0.85"]
    d_est = D.h - 4.0 - dbe - D.db_long / 2
    lat0.append(rf"d \approx h - r - d_{{be}} - \frac{{d_b}}{{2}} = {D.h:g} - 4 - {dbe:.2f} - \frac{{{D.db_long:.2f}}}{{2}} = {d_est:.1f}\ \text{{cm}}"
                + (rf" \ge d = {d:g}\ \text{{cm}}\;\checkmark" if d <= d_est + 1.0 else rf" < d = {d:g}\ \text{{cm}}"))
    R.paso("1. Propiedades de la sección y del refuerzo", lat0,
           f"Av es el área de todas las ramas verticales del estribo que cruzan una fisura diagonal "
           f"(estribo Ø{D.estribo} de {D.ramas} ramas). φ = 0.85 es el factor de reducción para corte de la E.060. "
           f"Se comprueba que el peralte efectivo d sea coherente con un recubrimiento libre de 4 cm, el estribo y "
           f"la barra longitudinal Ø = {D.db_long:.2f} cm.")
    if d > d_est + 1.0:
        R.alertas.append(f"El peralte efectivo d = {d:g} cm es mayor que h − r − Øe − db/2 = {d_est:.1f} cm: "
                         "revise d, porque un d mayor sobrestima la resistencia.")

    # ---------------- 1. Cortante último ----------------
    if D.modo == 1:
        Vap, w = D.wu * D.L / 2, D.wu
        R.x_fin = D.L * 50
        R.paso("2. Cortante último",
               [rf"R = V_u = \frac{{w_u L}}{{2}} = \frac{{{D.wu:g}\times{D.L:g}}}{{2}} = {Vap:.2f}\ \text{{t}}",
                rf"V_u(x) = w_u\left(\frac{{L}}{{2}} - x\right) = {D.wu:g}\left({D.L / 2:g} - x\right)",
                rf"M_u^{{máx}} = \frac{{w_u L^2}}{{8}} = \frac{{{D.wu:g}\times{D.L:g}^2}}{{8}} = {D.wu * D.L ** 2 / 8:.2f}\ \text{{t·m}}"],
               "En una viga simplemente apoyada con carga uniforme cada reacción vale la mitad de la carga total. "
               "El cortante es máximo en los apoyos y se anula en el centro de la luz, donde el momento es máximo.")
    elif D.modo == 2:
        wc = 1.25 * (D.CM + D.CV)
        wg = 1.4 * D.CM + 1.7 * D.CV
        a1 = D.As_sup * D.fy / (0.85 * D.fc * bw)
        a2 = D.As_inf * D.fy / (0.85 * D.fc * bw)
        M1 = D.As_sup * D.fy * (d - a1 / 2) / 1e5
        M2 = D.As_inf * D.fy * (d - a2 / 2) / 1e5
        Vcap = (M1 + M2) / D.ln + wc * D.ln / 2
        Vg = wg * D.ln / 2
        if Vcap >= Vg:
            Vap, w, ctrl = Vcap, wc, "Controla el diseño por capacidad."
        else:
            Vap, w, ctrl = Vg, wg, "Controla la combinación de gravedad."
        R.x_fin = D.ln * 50
        R.paso("2. Momentos nominales en los extremos",
               [rf"a^- = \frac{{A_s^- f_y}}{{0.85 f'_c b_w}} = \frac{{{D.As_sup:.2f}\times{D.fy:g}}}{{0.85\times{D.fc:g}\times{bw:g}}} = {a1:.2f}\ \text{{cm}}",
                rf"M_n^- = A_s^- f_y\left(d-\frac{{a^-}}{{2}}\right) = {D.As_sup:.2f}\times{D.fy:g}\left({d:g}-\frac{{{a1:.2f}}}{{2}}\right) = {M1:.2f}\ \text{{t·m}}",
                rf"a^+ = \frac{{{D.As_inf:.2f}\times{D.fy:g}}}{{0.85\times{D.fc:g}\times{bw:g}}} = {a2:.2f}\ \text{{cm}}",
                rf"M_n^+ = {D.As_inf:.2f}\times{D.fy:g}\left({d:g}-\frac{{{a2:.2f}}}{{2}}\right) = {M2:.2f}\ \text{{t·m}}"],
               "En el sismo se forman rótulas en los extremos: en un extremo trabaja el acero negativo y en el otro "
               "el positivo. Los momentos nominales se calculan con el acero realmente colocado.")
        R.paso("3. Cortante último por capacidad",
               [rf"w_u = 1.25(CM+CV) = 1.25({D.CM:g}+{D.CV:g}) = {wc:.3f}\ \text{{t/m}}",
                rf"V_u = \frac{{M_n^- + M_n^+}}{{l_n}} + \frac{{w_u l_n}}{{2}} = \frac{{{M1:.2f}+{M2:.2f}}}{{{D.ln:g}}} + \frac{{{wc:.3f}\times{D.ln:g}}}{{2}} = {(M1 + M2) / D.ln:.2f} + {wc * D.ln / 2:.2f} = {Vcap:.2f}\ \text{{t}}",
                rf"w_u^{{grav}} = 1.4\,CM + 1.7\,CV = 1.4({D.CM:g}) + 1.7({D.CV:g}) = {wg:.2f}\ \text{{t/m}}",
                rf"V_u^{{grav}} = \frac{{w_u l_n}}{{2}} = \frac{{{wg:.2f}\times{D.ln:g}}}{{2}} = {Vg:.2f}\ \text{{t}}",
                rf"V_u = \max({Vcap:.2f};\ {Vg:.2f}) = {Vap:.2f}\ \text{{t}}"],
               "El cortante de diseño es el que equilibra los momentos nominales más la carga de gravedad. Así la "
               "viga falla primero por flexión (dúctil) y no por corte (frágil). " + ctrl)
    elif D.modo == 3:
        Vap, w = D.Vu_cara, D.Vu_cara / D.luz_mitad
        R.x_fin = D.luz_mitad * 100
        R.paso("2. Cortante último",
               [rf"V_u^{{cara}} = {Vap:.2f}\ \text{{t}}",
                rf"w = \frac{{V_u^{{cara}}}}{{x_0}} = \frac{{{Vap:.2f}}}{{{D.luz_mitad:g}}} = {w:.3f}\ \text{{t/m}}",
                rf"V_u(x) = {Vap:.2f} - {w:.3f}\,x"],
               "El diagrama de cortante obtenido del análisis se aproxima por una recta desde la cara del apoyo "
               "hasta el punto de cortante nulo.")
    else:
        Vap, w = D.Vu_sec, 0.0
        R.x_fin = 0
        R.paso("2. Solicitaciones en la sección",
               [rf"V_u = {Vap:.2f}\ \text{{t}};\qquad N_u = {D.Nu:+.2f}\ \text{{t}}"],
               "Nu positivo es compresión y negativo es tracción. La carga axial modifica el aporte del concreto.")

    Vu = (lambda x: max(0.0, Vap - w * x / 100)) if D.modo != 4 else (lambda x: Vap)
    R.Vu = Vu
    Vud = Vu(d)
    if D.modo != 4:
        R.paso("   Sección crítica a la distancia d",
               [rf"V_u(d) = V_u - w\,d = {Vap:.2f} - {w:.3f}\times{d / 100:.2f} = {Vud:.2f}\ \text{{t}}"],
               "Cuando la reacción comprime el apoyo y no hay cargas concentradas cerca, la primera fisura diagonal "
               "nace a una distancia d de la cara; por eso se diseña con el cortante en esa sección.")

    # ---------------- Vc ----------------
    if D.Nu > 0:
        fac = 1 + D.Nu * 1000 / (140 * Ag)
        ftex = rf"\left(1+\frac{{N_u}}{{140A_g}}\right) = 1+\frac{{{D.Nu * 1000:,.0f}}}{{140\times{Ag:g}}} = {fac:.4f}"
    elif D.Nu < 0:
        fac = max(0.0, 1 + D.Nu * 1000 / (35 * Ag))
        ftex = rf"\left(1+\frac{{N_u}}{{35A_g}}\right) = 1-\frac{{{-D.Nu * 1000:,.0f}}}{{35\times{Ag:g}}} = {fac:.4f}"
    else:
        fac, ftex = 1.0, ""
    Vc = 0.53 * fac * D.lam * sq * bw * d / 1000
    phiVc = PHI * Vc
    R.phiVc = phiVc
    lat = []
    vck = f"{Vc * 1000:,.0f}".replace(",", r"\,")
    if ftex:
        lat.append(ftex.replace(",", r"\,"))
    lat.append(rf"V_c = 0.53{r'\,(\ldots)' if ftex else ''}\,\lambda\sqrt{{f'_c}}\,b_w d = 0.53{f'\\times{fac:.4f}' if ftex else ''}\times{D.lam:g}\times{sq:.3f}\times{bw:g}\times{d:g} = {vck}\ \text{{kg}} = {Vc:.2f}\ \text{{t}}")
    lat.append(rf"\phi V_c = 0.85\times{Vc:.2f} = {phiVc:.2f}\ \text{{t}};\qquad \frac{{\phi V_c}}{{2}} = {phiVc / 2:.2f}\ \text{{t}}")
    R.paso("4. Cortante que resiste el concreto", lat,
           "Vc agrupa lo que aporta el concreto: la zona comprimida, la trabazón de agregados y el efecto dovela. "
           + ("Con compresión axial Vc aumenta; con tracción disminuye." if ftex else ""))

    corte_total = D.modo == 4 and D.Nu < 0 and D.corte_total
    Vc_dis = 0.0 if corte_total else Vc
    if corte_total:
        R.paso("   Tracción axial significativa",
               [r"V_c = 0 \;\Rightarrow\; \text{los estribos toman el corte total}"],
               "Con tracción importante el concreto se fisura en todo el peralte y no se cuenta su aporte.")

    # ---------------- Clasificación ----------------
    if Vud > phiVc:
        cl_l = rf"V_u(d) = {Vud:.2f} > \phi V_c = {phiVc:.2f}\ \text{{t}} \;\Rightarrow\; \text{{estribos por cálculo}}"
        cl_t = "El concreto solo no alcanza: los estribos deben tomar el exceso Vu − φVc."
    elif Vud > phiVc / 2:
        cl_l = rf"\frac{{\phi V_c}}{{2}} = {phiVc / 2:.2f} < V_u(d) = {Vud:.2f} \le \phi V_c = {phiVc:.2f}\ \text{{t}} \;\Rightarrow\; \text{{estribo mínimo}}"
        cl_t = "El concreto resiste el corte, pero la norma exige el área mínima de estribos."
    else:
        cl_l = rf"V_u(d) = {Vud:.2f} \le \frac{{\phi V_c}}{{2}} = {phiVc / 2:.2f}\ \text{{t}} \;\Rightarrow\; \text{{sin estribos por cálculo}}"
        cl_t = "No se requieren estribos por cálculo; se colocan estribos de montaje."
    if corte_total:
        cl_l = rf"V_c = 0 \;\Rightarrow\; \text{{estribos por cálculo para}}\ V_u = {Vud:.2f}\ \text{{t}}"
        cl_t = "Toda la fuerza cortante la toman los estribos."
    R.paso("5. ¿Se necesitan estribos?", [cl_l], cl_t)

    # ---------------- Verificación de sección ----------------
    Vs_d = max(0.0, Vud / PHI - Vc_dis)
    Vsmax = 2.1 * sq * bw * d / 1000
    Vs11 = 1.1 * sq * bw * d / 1000
    secc_ok = Vs_d <= Vsmax
    R.paso("6. Verificación de las dimensiones de la sección",
           [rf"V_u \le \phi(V_c + V_s) \;\Rightarrow\; V_s = \frac{{V_u}}{{\phi}} - V_c = \frac{{{Vud:.2f}}}{{0.85}} - {Vc_dis:.2f} = {Vs_d:.2f}\ \text{{t}}",
            rf"V_{{s,máx}} = 2.1\sqrt{{f'_c}}\,b_w d = 2.1\times{sq:.3f}\times{bw:g}\times{d:g} = {Vsmax:.2f}\ \text{{t}}",
            rf"V_s = {Vs_d:.2f} " + (r"\le" if secc_ok else ">") + rf" {Vsmax:.2f}\ \text{{t}} \;\Rightarrow\; " + (r"\text{la sección es adecuada}" if secc_ok else r"\text{agrandar la sección}"),
            rf"1.1\sqrt{{f'_c}}\,b_w d = {Vs11:.2f}\ \text{{t}};\quad V_s = {Vs_d:.2f} " + (">" if Vs_d > Vs11 else r"\le") + rf" {Vs11:.2f} \;\Rightarrow\; s_{{máx}} = " + (r"d/4" if Vs_d > Vs11 else r"d/2")],
           "Si Vs supera 2.1√f′c·bw·d, el alma de concreto se aplastaría antes de que los estribos fluyan: no "
           "sirve agregar más estribos y hay que agrandar la sección. El límite 1.1√f′c·bw·d define si el "
           "espaciamiento máximo es d/2 o d/4.")
    R.ver["Vs ≤ 2.1√f′c·bw·d (aplastamiento del alma)"] = (Vs_d, Vsmax)
    if not secc_ok:
        R.ok = False
        R.alertas.append("La sección no alcanza: Vs > 2.1√f′c·bw·d. Aumente bw, h o f′c.")
        R.metr = dict(Vud=Vud, phiVc=phiVc, Vs=Vs_d, s=None)
        return R
    lim_d, lim_abs = (d / 4, 30) if Vs_d > Vs11 else (d / 2, 60)

    # ---------------- s máx y Av mín ----------------
    s1m = Av * D.fy / (0.2 * sq * bw)
    s2m = Av * D.fy / (3.5 * bw)
    smax = rd(min(lim_d, lim_abs, s1m, s2m), D.redondeo)
    R.smax = smax
    avmin1 = 0.2 * sq * bw * smax / D.fy
    avmin2 = 3.5 * bw * smax / D.fy
    R.paso("7. Espaciamiento máximo y acero mínimo",
           [rf"s \le {'d/4' if lim_d == d / 4 else 'd/2'} = {lim_d:.2f}\ \text{{cm}};\qquad s \le {lim_abs}\ \text{{cm}}",
            rf"s \le \frac{{A_v f_y}}{{0.2\sqrt{{f'_c}}\,b_w}} = \frac{{{Av:.3f}\times{D.fy:g}}}{{0.2\times{sq:.3f}\times{bw:g}}} = {s1m:.2f}\ \text{{cm}}",
            rf"s \le \frac{{A_v f_y}}{{3.5\,b_w}} = \frac{{{Av:.3f}\times{D.fy:g}}}{{3.5\times{bw:g}}} = {s2m:.2f}\ \text{{cm}}",
            rf"s_{{máx}} = \min({lim_d:.2f};\ {lim_abs};\ {s1m:.2f};\ {s2m:.2f}) \Rightarrow s_{{máx}} = {smax:g}\ \text{{cm}}",
            rf"A_{{v,mín}} = \max\left(0.2\sqrt{{f'_c}}\frac{{b_w s}}{{f_y}};\ 3.5\frac{{b_w s}}{{f_y}}\right) = \max({avmin1:.2f};\ {avmin2:.2f}) = {max(avmin1, avmin2):.2f} \le A_v = {Av:.2f}\ \text{{cm}}^2"],
           "Con s ≤ d/2 toda fisura posible a 45° queda cosida por al menos un estribo. El acero mínimo evita "
           "una falla brusca apenas se fisura el concreto.")

    def s_req(x):
        vs = Vu(max(x, d)) / PHI - Vc_dis
        return 1e9 if vs <= 1e-9 else Av * D.fy * d / (vs * 1000)
    R.s_req = s_req

    # ---------------- Zonas ----------------
    if D.modo != 4 and w > 0:
        Xc = (Vap - phiVc) / w * 100 if Vap > phiVc else 0.0
        Xm = (Vap - phiVc / 2) / w * 100 if Vap > phiVc / 2 else 0.0
        R.Xc, R.Xm = Xc, Xm
        txt = (f"De 0 a {Xc / 100:.2f} m los estribos se calculan; de {Xc / 100:.2f} a {Xm / 100:.2f} m basta el "
               f"estribo mínimo; desde {Xm / 100:.2f} m hasta el centro no se requieren por cálculo."
               if Xm < R.x_fin else
               f"Vu > φVc/2 en toda la viga: estribo mínimo en todo el tramo (Xm > {R.x_fin / 100:.2f} m).")
        R.paso("8. Zonas de la viga (desde la cara del apoyo)",
               [rf"X_c = \frac{{V_u - \phi V_c}}{{w}} = \frac{{{Vap:.2f}-{phiVc:.2f}}}{{{w:.3f}}} = {Xc / 100:.2f}\ \text{{m}}",
                rf"X_m = \frac{{V_u - \phi V_c/2}}{{w}} = \frac{{{Vap:.2f}-{phiVc / 2:.2f}}}{{{w:.3f}}} = {Xm / 100:.2f}\ \text{{m}}"], txt)

    # ---------------- s por cálculo ----------------
    s_d = s_req(d)
    if Vud > (0 if corte_total else phiVc):
        den = Vud * 1000 - (0 if corte_total else phiVc * 1000)
        R.paso("9. Espaciamiento por cálculo en la sección crítica",
               [r"V_s = \frac{A_v f_y d}{s} \;\Rightarrow\; s = " + (r"\frac{\phi A_v f_y d}{V_u}" if corte_total else r"\frac{\phi A_v f_y d}{V_u - \phi V_c}"),
                (r"s = \frac{\phi A_v f_y d}{V_u}" if corte_total else r"s = \frac{\phi A_v f_y d}{V_u - \phi V_c}") + rf" = \frac{{0.85\times{Av:.3f}\times{D.fy:g}\times{d:g}}}{{{den:,.0f}}} = {s_d:.2f}\ \text{{cm}}".replace(",", r"\,")],
               "Cada estribo que cruza la fisura aporta Av·fy; en una longitud d caben d/s estribos. Igualando "
               "esa fuerza al exceso de corte se despeja la separación s.")
    elif Vud > phiVc / 2:
        R.paso("9. Espaciamiento por cálculo", [rf"\phi V_c/2 < V_u \le \phi V_c \;\Rightarrow\; s = s_{{máx}} = {smax:g}\ \text{{cm}}"],
               "El concreto resiste el corte, pero se coloca el estribo mínimo @ s máx.")
    else:
        R.paso("9. Espaciamiento por cálculo", [rf"V_u \le \phi V_c/2"],
               "No se requieren estribos por cálculo; solo de montaje.")

    # ---------------- Distribución ----------------
    if D.modo == 4:
        s_us = min(rd(s_d, D.redondeo), smax)
        R.primero = 0
        R.tramos = [("todo", s_us)]
        R.distribucion = f"Ø{D.estribo} ({Av:.2f} cm²) @ {s_us:g} cm"
        lat = [rf"s = \min(s_{{cálculo}};\ s_{{máx}}) = \min({min(s_d, 999):.2f};\ {smax:g}) \Rightarrow \text{{usar}}\ s = {s_us:g}\ \text{{cm}}"]
        txt = "Se redondea hacia abajo a un valor práctico de obra."
        if D.s_exist:
            okx = D.s_exist <= s_us
            lat.append(rf"s_{{existente}} = {D.s_exist:g}\ \text{{cm}}\ " + (r"\le" if okx else ">") + rf"\ {s_us:g}\ \text{{cm}} \;\Rightarrow\; " + (r"\text{cumple}" if okx else r"\text{no cumple}"))
            t2 = (f"Estribos existentes @ {D.s_exist:g} cm: " +
                  ("CUMPLEN." if okx else f"NO CUMPLEN, superan s = {s_us:g} cm."))
            txt += " " + t2
            if not okx:
                R.alertas.append(t2)
        R.paso("10. Espaciamiento a usar", lat, txt)
        if D.ln_col > 0:
            dmin, dmax = min(bw, D.h), max(bw, D.h)
            Lo = max(D.ln_col * 100 / 6, dmax, 50)
            s0 = min(8 * D.db_long, dmin / 2, 10)
            s0u = max(2.5, rd(min(s0, s_us), 2.5))
            n0 = math.ceil((Lo - 5) / s0u - 1e-9)
            sres = min(s_us, 16 * D.db_long, 48 * dbe, dmin, 30)
            sc = rd(sres, D.redondeo)
            R.conf_col = dict(Lo=Lo, s0=s0u, n0=n0, sc=sc)
            R.tramos = [("todo", sc)]
            R.distribucion = (f"□ Ø{D.estribo}: 1 @ 5, {n0} @ {s0u:g}, resto @ {sc:g} cm "
                              f"(desde cada extremo; Lo = {Lo:.0f} cm)")
            R.ver["Separación en la zona de confinamiento"] = (s0u, s0)
            R.paso("11. Confinamiento de la columna (E.060, 21.4.5)",
                   [rf"L_o \ge \max\left(\frac{{l_n}}{{6}};\ \text{{mayor dimensión}};\ 50\right) = \max\left(\frac{{{D.ln_col * 100:g}}}{{6}};\ {dmax:g};\ 50\right) = {Lo:.0f}\ \text{{cm}}",
                    rf"s_0 \le \min\left(8d_b = {8 * D.db_long:.1f};\ \frac{{b_{{mín}}}}{{2}} = {dmin / 2:.1f};\ 10\right) = {s0:.1f}\ \text{{cm}} \Rightarrow @\ {s0u:g}\ \text{{cm}}",
                    rf"n = \frac{{L_o - 5}}{{s_0}} = \frac{{{Lo:.0f} - 5}}{{{s0u:g}}} \Rightarrow {n0}\ \text{{estribos}}",
                    rf"\text{{Fuera de}}\ L_o:\ s \le \min(s_{{corte}} = {s_us:g};\ 16d_b = {16 * D.db_long:.1f};\ 48d_{{be}} = {48 * dbe:.1f};\ b_{{mín}} = {dmin:g};\ 30) \Rightarrow @\ {sc:g}\ \text{{cm}}"],
                   "En los extremos de la columna se pueden formar rótulas durante un sismo: ahí los estribos van más "
                   "juntos para confinar el núcleo de concreto y evitar que las barras longitudinales pandeen. Fuera "
                   "de Lo manda el mayor entre el corte y el arriostramiento de las barras.")
    elif D.modo == 2:
        L0 = 2 * D.h
        s0 = min(d / 4, 10 * D.db_long, 24 * dbe, 30)
        r5 = max(D.redondeo, 5)
        s0u = rd(min(s0, s_d), r5) or rd(min(s0, s_d), 2.5)
        n0 = math.ceil((L0 - 5) / s0u - 1e-9)
        Vu2 = Vu(L0)
        vs2 = max(0.0, Vu2 / PHI - Vc_dis)
        s2 = Av * D.fy * d / (vs2 * 1000) if vs2 > 0 else 1e9
        lim2 = d / 4 if vs2 > Vs11 else d / 2
        sf = rd(min(s2, lim2, s1m, s2m, 60), r5)
        R.smax_resto = min(lim2, s1m, s2m, 60)
        R.primero, R.tramos, R.zona_conf = 5.0, [(n0, s0u), ("resto", sf)], L0
        R.ver["Separación en la zona de confinamiento 2h"] = (s0u, s0)
        R.distribucion = f"□ Ø{D.estribo}: 1 @ 0.05, {n0} @ {s0u / 100:.2f}, resto @ {sf / 100:.2f} m"
        R.paso("10. Zona de confinamiento (2h desde cada cara)",
               [rf"L_o = 2h = 2\times{D.h:g} = {L0:g}\ \text{{cm}}",
                rf"s_0 \le \min\left(\frac{{d}}{{4}}={d / 4:.1f};\ 10d_b={10 * D.db_long:.1f};\ 24d_{{be}}={24 * dbe:.1f};\ 30\right) = {s0:.1f}\ \text{{cm}}",
                rf"s_0 = \min({s0:.1f};\ s_{{cálculo}}={min(s_d, 999):.1f}) \Rightarrow @\ {s0u:g}\ \text{{cm}}",
                rf"n = \frac{{L_o - 5}}{{s_0}} = \frac{{{L0:g} - 5}}{{{s0u:g}}} = {(L0 - 5) / s0u:.1f} \Rightarrow {n0}\ \text{{estribos}}"],
               "En los extremos se forman las rótulas plásticas: los estribos cerrados y cercanos confinan el concreto "
               "y evitan el pandeo de las barras. El primer estribo va a 5 cm de la cara (máximo 10 cm). Límites de "
               "la E.060 21.4.4 (muros y dual tipo I); en pórticos y dual tipo II (21.5) el límite es 8db en lugar de 10db.")
        R.paso("11. Fuera de la zona de confinamiento",
               [rf"V_u(2h) = {Vap:.2f} - {w:.3f}\times{L0 / 100:.2f} = {Vu2:.2f}\ \text{{t}}",
                rf"V_s = \frac{{{Vu2:.2f}}}{{0.85}} - {Vc_dis:.2f} = {vs2:.2f}\ \text{{t}}",
                rf"s = \frac{{A_v f_y d}}{{V_s}} = \frac{{{Av:.3f}\times{D.fy:g}\times{d:g}}}{{{vs2 * 1000:,.0f}}} = {min(s2, 999):.1f}\ \text{{cm}}".replace(",", r"\,"),
                rf"s = \min({min(s2, 999):.1f};\ d/2 = {lim2:.1f}) \Rightarrow @\ {sf:g}\ \text{{cm}}"],
               "Fuera de 2h se diseña con el cortante en esa sección, respetando el espaciamiento máximo d/2.")
    else:
        if Vud <= phiVc / 2:
            R.primero, R.tramos = 5.0, [("resto", smax)]
            R.distribucion = f"Ø{D.estribo} de montaje: 1 @ 5, resto @ {smax:g} cm"
        else:
            s1 = min(rd(s_d, D.redondeo), smax)
            pr = max(5.0, rd(s1 / 2, 2.5))
            x_lim = max(min(R.Xm, R.x_fin), pr + smax)
            tr, x = [], pr
            while x < x_lim - 1e-6:
                s = min(rd(s_req(x), D.redondeo), smax)
                if s >= smax:
                    n = max(1, math.floor((x_lim - x) / smax + 1e-9))
                    tr.append((n, smax)); x += n * smax
                    break
                obj, n = min(s + D.paso, smax), 0
                while True:
                    n += 1
                    xn = x + n * s
                    if xn >= x_lim or (n >= D.min_estribos and s_req(xn) >= obj):
                        break
                tr.append((n, s)); x += n * s
            R.primero, R.tramos = pr, tr
            R.distribucion = f"Ø{D.estribo}: 1 @ {pr:g}, " + ", ".join(f"{n} @ {s:g}" for n, s in tr) + " cm (desde cada apoyo)"
        lat, pos = [], R.primero
        for n, s in R.tramos:
            if n == "resto":
                lat.append(rf"\text{{resto}}\ @\ {s:g}\ \text{{cm}}")
                break
            vu_x = Vu(max(pos, d))
            sr = s_req(pos)
            if sr < 900:
                lat.append(rf"x = {pos:.1f}:\ V_u = {vu_x:.2f}\ \text{{t}},\ s = \frac{{0.85\times{Av:.3f}\times{D.fy:g}\times{d:g}}}{{{(vu_x - (0 if corte_total else phiVc)) * 1000:,.0f}}} = {sr:.1f}\ \text{{cm}} \Rightarrow {n}\ @\ {s:g}\ \text{{cm}}".replace(",", r"\,"))
            else:
                lat.append(rf"x = {pos:.1f}:\ V_u = {vu_x:.2f} \le \phi V_c \Rightarrow s_{{máx}} \Rightarrow {n}\ @\ {s:g}\ \text{{cm}}")
            pos += n * s
        lat.append(rf"\text{{Último estribo a}}\ {pos:.1f}\ \text{{cm}};\ X_m = {R.Xm:.1f}\ \text{{cm}}")
        R.paso("10. Distribución escalonada de estribos", lat,
               f"Como el cortante disminuye hacia el centro, la separación puede crecer. En cada cambio de tramo se "
               f"recalcula s con el Vu de esa sección. Primer estribo a {R.primero:g} cm (≈ s/2). Se cambia de "
               f"espaciamiento cuando s sube {D.paso:g} cm o más, con al menos {D.min_estribos} estribos por tramo.")

    # ---------------- Detalle del estribo ----------------
    M_ = metrado(D, R, 4.0)
    R.paso("98. Detalle y metrado del estribo",
           [(rf"g = \max(8d_{{be}};\ 7.5) = \max(8\times{dbe:.2f};\ 7.5) = {M_['g']:.1f}\ \text{{cm}}\quad\text{{(gancho sísmico a 135}}^\circ\text{{)}}"
             if gancho(D)[2] else
             rf"g = 6d_{{be}} = 6\times{dbe:.2f} = {M_['g']:.1f}\ \text{{cm}}\quad\text{{(gancho a 135}}^\circ\text{{)}}"),
            rf"L_c = 2(b_w - 2r) + 2(h - 2r) + 2g = 2({bw:g} - 8) + 2({D.h:g} - 8) + 2({M_['g']:.1f}) = {M_['Lc']:.1f}\ \text{{cm}}",
            rf"n = {M_['n']}\ \text{{estribos ({M_['por']})}};\quad \text{{peso}} = {M_['n']}\times{M_['Lc'] / 100:.3f}\times{M_['peso']:.3f} = {M_['kg']:.2f}\ \text{{kg}}"],
           "Los estribos son cerrados, con ganchos a 135° que se anclan en el núcleo de concreto y se alternan en "
           "las esquinas. Un gancho a 90° se abre cuando el recubrimiento salta en un sismo. La longitud de corte y "
           "el peso sirven para el pedido de acero en obra.")

    # ---------------- Verificación ----------------
    lat, pos, okt = [], R.primero, True
    peor = (0.0, 1.0)
    for n, s in R.tramos:
        vu = Vu(max(pos, d))
        Vs_c = Av * D.fy * d / s / 1000
        pVn = PHI * (Vc_dis + Vs_c)
        ok = pVn >= vu - 1e-6
        okt &= ok
        if vu / pVn > peor[0] / peor[1]:
            peor = (vu, pVn)
        lat.append(rf"s={s:g}:\ V_s = \frac{{{Av:.3f}\times{D.fy:g}\times{d:g}}}{{{s:g}}} = {Vs_c:.2f};\ \phi V_n = 0.85({Vc_dis:.2f}+{Vs_c:.2f}) = {pVn:.2f} \ge V_u = {vu:.2f}\ \text{{t}}\ " + (r"\checkmark" if ok else r"\times"))
        if n in ("resto", "todo"):
            break
        pos += n * s
    R.paso("99. Verificación final φVn ≥ Vu", lat,
           ("Todos los tramos cumplen: la resistencia de diseño supera al cortante último en cada tramo."
            if okt else "Hay tramos que no cumplen."))
    R.ok = okt
    s_col = max(s for _, s in R.tramos)
    R.ver["Separación usada ≤ s máx"] = (s_col, getattr(R, "smax_resto", smax) if D.modo == 2 else smax)
    R.ver["Av ≥ Av,mín"] = (max(avmin1, avmin2), Av)
    R.ver["φVn ≥ Vu (tramo más exigido)"] = peor
    # aviso de viga de gran peralte: ahí manda el modelo de bielas (puntal-tensor), no los estribos
    luz = {1: D.L * 100, 2: D.ln * 100, 3: D.luz_mitad * 200}.get(D.modo)
    if luz and luz / D.h <= 4:
        R.alertas.append(f"Viga de gran peralte (luz/h = {luz / D.h:.1f} ≤ 4): la carga baja por bielas de "
                         "concreto; debe revisarse con el modelo puntal-tensor y no solo con estribos.")
    R.metr = dict(Vud=Vud, phiVc=phiVc, Vs=Vs_d, s=s_d, smax=smax)
    okall = R.ok and all(dm <= cp + 1e-6 for dm, cp in R.ver.values())
    R.paso("100. Respuesta",
           [r"\text{" + R.distribucion.replace('"', "''").replace("□ ", "").replace("Ø", "Ø") + "}"],
           ("Cumplen todas las verificaciones: resistencia φVn ≥ Vu en cada tramo, tamaño de la sección, "
            "espaciamiento máximo y acero mínimo." if okall else "Hay verificaciones que no cumplen: ver observaciones.")
           + (" Además, la viga es de gran peralte: complementar con el modelo puntal-tensor." if any("gran peralte" in a for a in R.alertas) else ""))
    R.texto.append("\nRESULTADO: " + R.distribucion)
    R.texto.append("\nREFERENCIAS")
    R.texto += ["  " + r.replace("*", "") for r in REFERENCIAS]
    return R


# ===========================================================================
# GRÁFICOS
# ===========================================================================
def figura(D, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if D.modo == 4:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        dbe, Ab = BARRAS[D.estribo]
        Av = D.Av_manual or D.ramas * Ab
        s_us = R.tramos[0][1]
        Vc_dis = 0 if (D.Nu < 0 and D.corte_total) else R.phiVc / PHI
        pVs = PHI * Av * D.fy * D.d / s_us / 1000
        etiquetas = ["Vu", "φVc", "φVs", "φVn = φVc + φVs"]
        valores = [R.Vu(0), PHI * Vc_dis, pVs, PHI * Vc_dis + pVs]
        colores = ["#C0392B", "#2E8B62", "#B0652A", "#0F2A52"]
        b = ax.barh(etiquetas[::-1], valores[::-1], color=colores[::-1])
        for rect, v in zip(b, valores[::-1]):
            ax.text(v, rect.get_y() + rect.get_height() / 2, f"  {v:.2f} t", va="center")
        ax.set_xlim(0, max(valores) * 1.25)
        ax.set_xlabel("Fuerza cortante (t)")
        ax.set_title(f"Resistencia de la sección con s = {s_us:g} cm", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    xf = R.x_fin
    x = np.linspace(0, xf, 400)
    vu = np.array([R.Vu(xi) for xi in x])
    fig, axs = plt.subplots(3, 1, figsize=(10, 10.5), gridspec_kw={"height_ratios": [3, 2.2, 1.5]})
    ax = axs[0]
    ax.plot(x / 100, vu, color="#0F2A52", lw=2.2, label="Vu")
    ax.fill_between(x / 100, vu, R.phiVc, where=vu > R.phiVc, color="#C0392B", alpha=0.25,
                    label="Exceso que toman los estribos")
    ax.axhline(R.phiVc, color="#2E8B62", ls="--", lw=1.6, label=f"φVc = {R.phiVc:.2f} t")
    ax.axhline(R.phiVc / 2, color="#B0652A", ls=":", lw=1.6, label=f"φVc/2 = {R.phiVc / 2:.2f} t")
    ax.axvline(D.d / 100, color="gray", lw=1)
    ax.text(D.d / 100, max(vu) * 1.02, " d", color="gray")
    for X, nom, yy in [(R.Xc, "Xc", 0.62), (R.Xm, "Xm", 0.48)]:
        if 0 < X < xf:
            ax.axvline(X / 100, color="black", lw=0.8, ls="-.")
            ax.text(X / 100, max(vu) * yy, f" {nom} = {X / 100:.2f} m",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    if R.zona_conf:
        ax.axvspan(0, R.zona_conf / 100, color="#6E4BB0", alpha=0.10, label="Confinamiento 2h")
    ax.set_xlim(0, xf / 100); ax.set_ylim(0, max(vu) * 1.12)
    ax.set_ylabel("Cortante (t)"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Diagrama de cortante (media viga)", fontweight="bold")

    ax = axs[1]
    sr = np.array([min(R.s_req(xi), 100) for xi in x])
    ax.plot(x / 100, sr, color="#B0652A", lw=2, label="s requerido")
    ax.axhline(R.smax, color="#2E8B62", ls="--", label=f"s máx = {R.smax:g} cm")
    xs, ys, pos = [], [], R.primero
    for n, s in R.tramos:
        fin = xf if n == "resto" else pos + n * s
        xs += [pos / 100, fin / 100]; ys += [s, s]; pos = fin
        if n == "resto":
            break
    ax.plot(xs, ys, color="#0F2A52", lw=2.6, label="s colocado")
    ax.set_xlim(0, xf / 100); ax.set_ylim(0, max(R.smax * 1.8, max(ys) * 1.3))
    ax.set_xlabel("Distancia desde la cara del apoyo (m)"); ax.set_ylabel("s (cm)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")

    ax = axs[2]
    H = D.h / 100
    ax.add_patch(Rectangle((0, 0), xf / 100, H, fill=False, lw=1.5))
    ax.add_patch(Rectangle((-0.25, -0.15), 0.25, H + 0.3, color="#BBBBBB"))
    cols = ["#C0392B", "#0F2A52", "#2E8B62", "#B0652A", "#6E4BB0", "#555555"]
    pos = R.primero
    ax.plot([pos / 100] * 2, [0.03, H - 0.03], color=cols[0], lw=1.5)
    for k, (n, s) in enumerate(R.tramos):
        cnt = int((xf - pos) // s) if n == "resto" else n
        c, ini = cols[k % len(cols)], pos
        for _ in range(cnt):
            pos += s
            ax.plot([pos / 100] * 2, [0.03, H - 0.03], color=c, lw=1.5)
        ax.text((ini + pos) / 200, -0.13, f"resto @ {s:g}" if n == "resto" else f"{n} @ {s:g}",
                ha="center", fontsize=9, color=c)
        if n == "resto":
            break
    ax.text(R.primero / 100, H + 0.04, f"1 @ {R.primero:g}", ha="center", fontsize=9, color=cols[0])
    ax.set_xlim(-0.35, xf / 100 + 0.1); ax.set_ylim(-0.25, H + 0.2); ax.axis("off")
    ax.set_title("Distribución de estribos (cm)", fontsize=11)
    fig.tight_layout()
    return fig


# ===========================================================================
# REPORTE PDF (memoria de cálculo para obra)
# ===========================================================================
PESO_BARRA = {"6 mm": 0.222, "8 mm": 0.395, "3/8\"": 0.560, "12 mm": 0.888,
              "1/2\"": 0.994, "5/8\"": 1.552, "3/4\"": 2.235, "1\"": 3.973}  # kg/m


def gancho(D):
    """Extensión del gancho a 135° del estribo: (g en cm, k veces db, es sísmico).
    Elementos sísmicos: gancho sísmico 8db ≥ 75 mm (E.060 21.1). Otros: 6db (E.060 7.1.3)."""
    dbe = BARRAS[D.estribo][0]
    sis = D.modo == 2 or (D.modo == 4 and D.ln_col > 0) or D.modo == 5
    return (max(8 * dbe, 7.5), 8, True) if sis else (6 * dbe, 6, False)


def longitud_viga(D):
    return {1: D.L * 100, 2: D.ln * 100, 3: D.luz_mitad * 200}.get(D.modo, 0.0)


def posiciones_estribos(D, R):
    """Posiciones (cm) de los estribos en media viga, desde la cara del apoyo.
    Devuelve (lista de (x, tipo)), con tipo 'calculo' o 'montaje'."""
    half = longitud_viga(D) / 2
    pos, x = [], R.primero
    pos.append((x, "calculo"))
    s_ult = R.smax
    for n, s in R.tramos:
        if n == "resto":
            while x + s <= half + 1e-6:
                x += s
                pos.append((x, "calculo"))
            s_ult = s
            break
        for _ in range(n):
            x += s
            pos.append((x, "calculo"))
        s_ult = s
    else:
        # zona central sin estribos por cálculo: montaje @ s máx
        while x + R.smax <= half + 1e-6:
            x += R.smax
            pos.append((x, "montaje"))
        s_ult = R.smax
    return pos, s_ult


def metrado(D, R, rec):
    dbe, _ = BARRAS[D.estribo]
    g = gancho(D)[0]
    Lc = 2 * (D.bw - 2 * rec) + 2 * (D.h - 2 * rec) + 2 * g
    peso = PESO_BARRA[D.estribo]
    if D.modo == 5:
        lm = D.lm_proy + D.col_b - rec     # proyección de la ménsula + tramo dentro de la columna
        Lc5 = 2 * (D.bw - 2 * rec) + 2 * (lm - 2 * rec) + 2 * g
        n = R.metr["ne"]
        return dict(g=g, Lc=Lc5, n=n, por="por ménsula", L=n * Lc5 / 100, kg=n * Lc5 / 100 * peso,
                    peso=peso, n_calc=n, n_mont=0)
    if D.modo == 4 and R.conf_col:
        cc = R.conf_col
        Lb = D.ln_col * 100
        x_end = 5 + cc["n0"] * cc["s0"]
        n = 2 * (cc["n0"] + 1) + max(0, math.ceil((Lb - 2 * x_end) / cc["sc"] - 1e-9) - 1)
        return dict(g=g, Lc=Lc, n=n, por="por columna", L=n * Lc / 100, kg=n * Lc / 100 * peso,
                    peso=peso, n_calc=n, n_mont=0)
    if D.modo == 4:
        s = R.tramos[0][1]
        n = math.ceil(100 / s)
        return dict(g=g, Lc=Lc, n=n, por="por metro lineal", L=n * Lc / 100, kg=n * Lc / 100 * peso,
                    peso=peso, n_calc=n, n_mont=0)
    pos, s_ult = posiciones_estribos(D, R)
    half = longitud_viga(D) / 2
    en_centro = abs(pos[-1][0] - half) < 1e-6   # estribo justo en el centro: no contarlo dos veces
    n = 2 * len(pos) - (1 if en_centro else 0)
    centro = (not en_centro) and 2 * (half - pos[-1][0]) > s_ult + 1e-6
    if centro:
        n += 1
    n_m = 2 * sum(1 for _, t in pos if t == "montaje") + (1 if centro else 0)
    return dict(g=g, Lc=Lc, n=n, por="por viga", L=n * Lc / 100, kg=n * Lc / 100 * peso,
                peso=peso, n_calc=n - n_m, n_mont=n_m)


def figura_detalle(D, R, rec):
    if D.modo == 5:
        return detalle_mensula(D, R, rec)
    """Elevación de la viga completa y sección transversal con el estribo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    dbe, _ = BARRAS[D.estribo]
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1])

    # ---- elevación
    ax = fig.add_subplot(gs[0])
    if D.modo == 4 and R.conf_col:
        cc = R.conf_col
        Lb = D.ln_col * 100
        xs = [5 + i * cc["s0"] for i in range(cc["n0"] + 1)]
        x_end = xs[-1]
        xm = list(np.arange(x_end + cc["sc"], Lb - x_end - cc["sc"] / 2, cc["sc"]))
        pos_all = [(x, "calculo") for x in xs] + [(Lb - x, "calculo") for x in xs] + [(x, "montaje") for x in xm]
        etiquetas = [(0, x_end, f"1@5, {cc['n0']}@{cc['s0']:g}"), (x_end, Lb - x_end, f"resto @ {cc['sc']:g}"),
                     (Lb - x_end, Lb, f"{cc['n0']}@{cc['s0']:g}, 1@5")]
    elif D.modo == 4:
        Lb = 300.0
        s = R.tramos[0][1]
        xs = list(np.arange(s / 2, Lb, s))
        pos_all = [(x, "calculo") for x in xs]
        etiquetas = [(0, Lb, f"Ø{D.estribo} @ {s:g} cm")]
    else:
        Lb = longitud_viga(D)
        pos, _ = posiciones_estribos(D, R)
        pos_all = pos + [(Lb - x, t) for x, t in pos]
        etiquetas, x0 = [], 0.0
        etiquetas.append((0, R.primero, f"1@{R.primero:g}"))
        x = R.primero
        for n, s in R.tramos:
            if n == "resto":
                etiquetas.append((x, Lb / 2, f"resto @ {s:g}"))
                break
            etiquetas.append((x, x + n * s, f"{n}@{s:g}"))
            x += n * s
        if pos and pos[-1][1] == "montaje":
            etiquetas.append((x, Lb / 2, f"montaje @ {R.smax:g}"))
    H = D.h
    ax.add_patch(Rectangle((0, 0), Lb, H, fill=False, lw=1.6))
    if D.modo == 4 and R.conf_col:
        for a_, b_ in ((0, R.conf_col["Lo"]), (Lb - R.conf_col["Lo"], Lb)):
            ax.add_patch(Rectangle((a_, 0), b_ - a_, H, facecolor="#6E4BB0", alpha=0.08))
        ax.text(R.conf_col["Lo"] / 2, H * 1.08, f"Lo = {R.conf_col['Lo']:.0f}", ha="center", fontsize=7.5, color="#6E4BB0")
        ax.text(Lb / 2, H * 1.3, f"Columna: ln = {Lb / 100:.2f} m (dibujada en horizontal)", ha="center", fontsize=8.5)
    if D.modo != 4:
        for xc in (-30, Lb):
            ax.add_patch(Rectangle((xc, -0.6 * H), 30, 2.2 * H, facecolor="#D5D8DC", edgecolor="#555", lw=1))
        ax.axvline(Lb / 2, color="#999", ls="-.", lw=0.8)
        ax.text(Lb / 2, H * 1.55, "CL", ha="center", fontsize=8, color="#555")
    ax.plot([0, Lb], [rec, rec], color="#333", lw=1.3)
    ax.plot([0, Lb], [H - rec, H - rec], color="#333", lw=1.3)
    for x, t in pos_all:
        ax.plot([x, x], [rec * 0.6, H - rec * 0.6], color="#C0392B" if t == "calculo" else "#2E86C1",
                lw=1.1, ls="-" if t == "calculo" else "--")
    yb = -0.35 * H
    for a, b, txt in etiquetas:
        if b - a <= 0:
            continue
        ax.annotate("", xy=(a, yb), xytext=(b, yb), arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333"))
        ax.text((a + b) / 2, yb - 0.17 * H, txt, ha="center", va="top", fontsize=7.5, rotation=0)
    if D.modo != 4:
        ax.annotate("", xy=(0, H * 1.3), xytext=(Lb, H * 1.3), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(Lb / 2 - Lb * 0.12, H * 1.38, f"{'L' if D.modo == 1 else 'ln'} = {Lb / 100:.2f} m", fontsize=8.5)
        if R.zona_conf:
            for a, b in ((0, R.zona_conf), (Lb - R.zona_conf, Lb)):
                ax.add_patch(Rectangle((a, 0), b - a, H, facecolor="#6E4BB0", alpha=0.08))
            ax.text(R.zona_conf / 2, H * 1.08, f"2h = {R.zona_conf:g}", ha="center", fontsize=7.5, color="#6E4BB0")
    ax.set_xlim(-45, Lb + 45)
    ax.set_ylim(-1.0 * H, 1.65 * H)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("ELEVACIÓN: distribución de estribos (cm, desde la cara de cada apoyo)", fontsize=10, fontweight="bold")

    # ---- sección
    ax = fig.add_subplot(gs[1])
    b, h, r = D.bw, D.h, rec
    ax.add_patch(Rectangle((0, 0), b, h, facecolor="#F2F3F4", edgecolor="black", lw=1.5))
    ax.add_patch(Rectangle((r, r), b - 2 * r, h - 2 * r, fill=False, edgecolor="#C0392B", lw=2))
    # gancho a 135° en la esquina superior izquierda
    L1 = gancho(D)[0]
    k = L1 / math.sqrt(2)
    ax.plot([r, r + k], [h - r, h - r - k], color="#C0392B", lw=2)
    ax.plot([r, r + k * 0.95], [h - r, h - r - k * 1.05], color="#C0392B", lw=2)
    db = max(D.db_long, 1.27)
    for cx in (r + dbe + db / 2, b - r - dbe - db / 2):
        for cy in (r + dbe + db / 2, h - r - dbe - db / 2):
            ax.add_patch(Circle((cx, cy), db / 2, color="black"))
    ax.annotate("", xy=(0, -4), xytext=(b, -4), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(b / 2, -9, f"bw = {b:g}", ha="center", fontsize=8.5)
    ax.annotate("", xy=(-4, 0), xytext=(-4, h), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-7, h / 2, f"h = {h:g}", ha="right", va="center", rotation=90, fontsize=8.5)
    ax.text(b + 3, h - r - k / 2, f"Gancho 135°\next. {L1:.1f} cm", fontsize=7, va="center")
    ax.text(b + 3, h / 2, f"Estribo\nØ{D.estribo}", fontsize=7.5, va="center", color="#C0392B")
    ax.text(b + 3, r / 2 + 2, f"r = {r:g}", fontsize=7, va="center")
    ax.set_xlim(-16, b + 22)
    ax.set_ylim(-14, h + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("SECCIÓN", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _latex_mathtext(l):
    import re
    l = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", l)
    l = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", l)
    return l.replace(r"\checkmark", r"\ \mathrm{OK}")


def _formula_img(latex, ancho_max_cm):
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    from matplotlib.mathtext import math_to_image
    from reportlab.lib.units import cm
    from reportlab.platypus import Image
    from PIL import Image as PILImage
    rcParams["mathtext.fontset"] = "cm"
    buf = io.BytesIO()
    math_to_image("$" + _latex_mathtext(latex) + "$", buf, dpi=220, format="png")
    buf.seek(0)
    w, h = PILImage.open(buf).size
    buf.seek(0)
    escala = 2.54 / 220 * 1.35  # cm por píxel
    wc, hc = w * escala, h * escala
    if wc > ancho_max_cm:
        hc *= ancho_max_cm / wc
        wc = ancho_max_cm
    return Image(buf, width=wc * cm, height=hc * cm)


def _fig_img(fig, ancho_cm):
    import io
    from reportlab.lib.units import cm
    from reportlab.platypus import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    buf.seek(0)
    w, h = fig.get_size_inches()
    from PIL import Image as PILImage
    pw, ph = PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=ancho_cm * cm, height=ancho_cm * ph / pw * cm)


def generar_pdf(D, R, info):
    """Genera la memoria de cálculo en PDF y devuelve los bytes."""
    import io
    import datetime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    PageBreak, KeepTogether)

    # Fuente con todos los símbolos (², √, φ, ≤): DejaVu Sans, incluida con matplotlib
    import os
    import matplotlib as _mpl
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    fdir = os.path.join(_mpl.get_data_path(), "fonts", "ttf")
    try:
        for nom, arch in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                          ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
            pdfmetrics.registerFont(TTFont(nom, os.path.join(fdir, arch)))
        addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B")
        addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")
        F, FB = "DV", "DV-B"
    except Exception:
        F, FB = "Helvetica", "Helvetica-Bold"
    AZUL = colors.HexColor("#0F2A52")
    GRIS = colors.HexColor("#6A7890")
    CLARO = colors.HexColor("#E8EEF6")
    ss = getSampleStyleSheet()
    st_n = ParagraphStyle("n", parent=ss["Normal"], fontName=F, fontSize=9, leading=12.5, alignment=TA_JUSTIFY)
    st_s = ParagraphStyle("s", parent=st_n, fontSize=8, leading=10.5)
    st_c = ParagraphStyle("c", parent=st_s, textColor=GRIS, alignment=2)
    st_h1 = ParagraphStyle("h1", parent=st_n, fontName=FB, fontSize=11.5, textColor=AZUL,
                           spaceBefore=10, spaceAfter=5)
    st_h2 = ParagraphStyle("h2", parent=st_n, fontName=FB, fontSize=9.8, textColor=AZUL,
                           spaceBefore=6, spaceAfter=2)
    st_t = ParagraphStyle("t", parent=st_n, fontName=FB, fontSize=14, alignment=TA_CENTER,
                          textColor=AZUL, leading=18)

    dbe, Ab = BARRAS[D.estribo]
    Av = D.Av_manual or D.ramas * Ab
    rec = info.get("rec", 4.0)
    fecha = info.get("fecha") or datetime.date.today().strftime("%d/%m/%Y")
    proyecto = info.get("proyecto") or "—"
    elemento = info.get("elemento") or D.titulo

    def tabla(filas, anchos, cab=True, zebra=True):
        t = Table([[Paragraph(str(c), st_s) for c in f] for f in filas], colWidths=[a * cm for a in anchos])
        estilo = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA8BC")),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if cab:
            estilo += [("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
            t2 = Table([[Paragraph(f"<font color='white'><b>{c}</b></font>", st_s) for c in filas[0]]] +
                       [[Paragraph(str(c), st_s) for c in f] for f in filas[1:]],
                       colWidths=[a * cm for a in anchos])
            t = t2
        if zebra:
            for i in range(1 if cab else 0, len(filas)):
                if i % 2 == 0:
                    estilo.append(("BACKGROUND", (0, i), (-1, i), CLARO))
        t.setStyle(TableStyle(estilo))
        return t

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(AZUL)
        canvas.setLineWidth(0.8)
        canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
        canvas.setFont(F, 7)
        canvas.setFillColor(GRIS)
        canvas.drawString(2 * cm, A4[1] - 1.3 * cm, f"MEMORIA DE CÁLCULO · DISEÑO POR CORTANTE · {elemento}"[:110])
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, proyecto[:60])
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.drawString(2 * cm, 1.1 * cm, "Norma E.060 Concreto Armado (RNE) · Ortega (2014), cap. 7")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"Memoria de cálculo - {elemento}", author=info.get("elaborado", ""))
    E = []
    _n = [0]

    def sec(t):
        _n[0] += 1
        return f"{_n[0]}. {t}"

    # ---------------- Carátula de datos ----------------
    E.append(Paragraph("MEMORIA DE CÁLCULO ESTRUCTURAL", st_t))
    E.append(Paragraph("DISEÑO POR FUERZA CORTANTE DE ELEMENTOS DE CONCRETO ARMADO",
                       ParagraphStyle("st", parent=st_t, fontSize=10.5, textColor=GRIS)))
    E.append(Spacer(1, 8))
    cab = [["Proyecto", proyecto], ["Elemento", elemento], ["Ubicación", info.get("ubicacion") or "—"],
           ["Normativa", "Norma E.060 Concreto Armado, Reglamento Nacional de Edificaciones (MVCS, 2009)"],
           ["Elaborado por", info.get("elaborado") or "—"], ["Revisado por", info.get("revisado") or "—"],
           ["Fecha", fecha]]
    t = Table([[Paragraph(f"<b>{a}</b>", st_s), Paragraph(b, st_s)] for a, b in cab], colWidths=[3.6 * cm, 13.4 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA8BC")),
                           ("BACKGROUND", (0, 0), (0, -1), CLARO), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    E.append(t)

    # resultado destacado
    ok = R.ok and not R.alertas
    M5 = D.modo == 5
    refs = REFERENCIAS + ([REF_PERLES, REF_ACI] if M5 else [])
    caja = Table([[Paragraph(f"<b>REFUERZO A COLOCAR:</b> {R.distribucion}", st_n)],
                  [Paragraph("<b>Estado:</b> " + ("CUMPLE con la Norma E.060" if ok else "REVISAR: ver observaciones"), st_s)]],
                 colWidths=[17 * cm])
    caja.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E3F4E8") if ok else colors.HexColor("#FDECEA")),
                              ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2E8B62") if ok else colors.HexColor("#C0392B")),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    E += [Spacer(1, 8), caja]

    # ---------------- Planteamiento ----------------
    if info.get("enunciado"):
        E.append(Paragraph(sec("PLANTEAMIENTO DEL PROBLEMA"), st_h1))
        E.append(Paragraph(info["enunciado"], st_n))
        if info.get("fuente"):
            E.append(Paragraph(f"<i>Fuente: {info['fuente']}.</i>", st_s))

    # ---------------- 1. Objetivo ----------------
    E.append(Paragraph(sec("OBJETIVO Y ALCANCE"), st_h1))
    if M5:
        E.append(Paragraph(
            f"Diseñar la ménsula corta <b>{elemento}</b>. En este elemento la carga llega muy cerca del apoyo y el "
            f"corte no se resiste con estribos verticales sino con una biela de concreto en compresión y un tensor "
            f"de acero (Perles, 2009, p. 192). Se determina el acero del tensor Asc y los estribos horizontales Ah, "
            f"y se verifica que la biela no se aplaste, según las disposiciones de la Norma E.060 para ménsulas.", st_n))
    else:
      E.append(Paragraph(
        f"Determinar el refuerzo transversal (estribos) del elemento <b>{elemento}</b> para resistir la fuerza "
        f"cortante última, verificando la resistencia de la sección, el espaciamiento máximo y el acero mínimo "
        f"según la Norma E.060. Tipo de análisis: <i>{MODOS[D.modo]}</i>. El diseño por flexión del acero "
        f"longitudinal no forma parte de esta memoria.", st_n))

    # ---------------- 2. Normativa ----------------
    E.append(Paragraph(sec("NORMATIVA Y BIBLIOGRAFÍA"), st_h1))
    for ref in refs:
        E.append(Paragraph("• " + ref.replace("*", ""), st_s))
    E.append(Paragraph("Método de diseño: resistencia última con φ = 0.85; " +
                       ("disposiciones de la E.060 para ménsulas (corte-fricción) y verificación por el método puntal-tensor del ACI 318-08, Apéndice A (φ = 0.75)." if M5 else
                        "φVn ≥ Vu, con Vn = Vc + Vs."), st_s))

    # ---------------- 3. Datos ----------------
    E.append(Paragraph(sec("DATOS DE DISEÑO"), st_h1))
    geo = [["Parámetro", "Símbolo", "Valor", "Unidad"],
           ["Ancho del alma", "bw", f"{D.bw:g}", "cm"], ["Peralte total", "h", f"{D.h:g}", "cm"],
           ["Peralte efectivo", "d", f"{D.d:g}", "cm"], ["Recubrimiento libre", "r", f"{rec:g}", "cm"]]
    if D.modo == 1:
        geo.append(["Luz de cálculo", "L", f"{D.L:g}", "m"])
    elif D.modo == 2:
        geo.append(["Luz libre", "ln", f"{D.ln:g}", "m"])
    elif D.modo == 3:
        geo.append(["Distancia de la cara a Vu = 0", "—", f"{D.luz_mitad:g}", "m"])
    elif M5:
        geo += [["Distancia de la carga a la cara de la columna", "a", f"{D.a_carga:g}", "cm"],
                ["Proyección de la ménsula", "lm", f"{D.lm_proy:g}", "cm"],
                ["Altura en el borde libre", "hb", f"{D.h_borde:g}", "cm"],
                ["Placa de apoyo (largo × ancho)", "lp × bp", f"{D.placa_l:g} × {D.placa_b:g}", "cm"],
                ["Dimensión de la columna (dirección del tensor)", "bcol", f"{D.col_b:g}", "cm"]]
    mat = [["Material", "Símbolo", "Valor", "Unidad"],
           ["Resistencia del concreto", "f′c", f"{D.fc:g}", "kg/cm²"],
           ["Fluencia del acero de estribos", "fy", f"{D.fy:g}", "kg/cm²"],
           ["Factor de concreto liviano", "λ", f"{D.lam:g}", "—"],
           ["Factor de reducción (corte)", "φ", "0.85", "—"]]
    car = [["Solicitación", "Símbolo", "Valor", "Unidad"]]
    if D.modo == 1:
        car.append(["Carga última uniforme", "wu", f"{D.wu:g}", "t/m"])
    elif D.modo == 2:
        car += [["Carga muerta", "CM", f"{D.CM:g}", "t/m"], ["Carga viva", "CV", f"{D.CV:g}", "t/m"],
                ["Acero negativo en el extremo", "As⁻", f"{D.As_sup:.2f}", "cm²"],
                ["Acero positivo en el extremo", "As⁺", f"{D.As_inf:.2f}", "cm²"],
                ["Menor diámetro longitudinal", "db", f"{D.db_long:g}", "cm"]]
    elif D.modo == 3:
        car.append(["Cortante en la cara del apoyo", "Vu", f"{D.Vu_cara:g}", "t"])
    else:
        car.append(["Cortante último en la sección", "Vu", f"{D.Vu_sec:g}", "t"])
    if M5:
        car.append(["Fuerza horizontal en el apoyo", "Nuc", f"{max(D.Nuc, 0.2 * D.Vu_sec):.2f}", "t"])
        ref_t = [["Refuerzo", "Símbolo", "Valor", "Unidad"],
                 ["Barras del tensor", "Ø", D.barra_ppal, "—"], ["Estribos horizontales", "Ø", D.estribo, "—"]]
    else:
        car.append(["Carga axial (+ compresión / − tracción)", "Nu", f"{D.Nu:g}", "t"])
        ref_t = [["Refuerzo transversal", "Símbolo", "Valor", "Unidad"],
                 ["Diámetro del estribo", "Ø", D.estribo, "—"], ["Número de ramas", "n", f"{D.ramas}", "—"],
                 ["Área de refuerzo por corte", "Av", f"{Av:.3f}", "cm²"]]
    anch = [7.5, 2.5, 3.5, 3.5]
    k = _n[0]
    for nombre, filas in ((f"{k}.1 Geometría", geo), (f"{k}.2 Materiales", mat), (f"{k}.3 Solicitaciones", car),
                          (f"{k}.4 Refuerzo transversal propuesto", ref_t)):
        E.append(KeepTogether([Paragraph(nombre, st_h2), tabla(filas, anch)]))

    # ---------------- 4. Cálculo ----------------
    E.append(Paragraph(sec("CÁLCULO DEL REFUERZO POR CORTANTE"), st_h1))
    kc = _n[0]
    for titulo, lat, txt in R.pasos:
        import re as _re
        tt = _re.sub(r"^(\d+)\. ", lambda mm: f"{kc}.{mm.group(1)} ", titulo.strip())
        bloque = [Paragraph(tt, st_h2)]
        filas = []
        for l, _ in lat:
            filas.append([_formula_img(l, 16.5)])
        if filas:
            tf = Table(filas, colWidths=[17 * cm])
            tf.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            bloque.append(tf)
        if txt:
            bloque.append(Paragraph(f"<i>{txt}</i>", st_s))
        E.append(KeepTogether(bloque))

    # ---------------- 5. Resumen ----------------
    m = R.metr
    E.append(Paragraph(sec("RESUMEN DE RESULTADOS"), st_h1))
    if M5:
        res = [["Concepto", "Valor", "Verificación"],
               ["Relación a/d", f"{D.a_carga / D.d:.2f}", "≤ 1: ménsula corta" if D.a_carga <= D.d else "> 1: no es ménsula corta"],
               ["Fuerza en el tensor, T", f"{m['T']:.2f} t", "modelo biela-tensor"],
               ["Fuerza en la biela, C", f"{m['C']:.2f} t", "concreto en compresión"],
               ["Vn requerido / límite de la biela", f"{m['Vn'] / 1000:.2f} / {m['lim'] / 1000:.2f} t",
                "CUMPLE" if m['Vn'] <= m['lim'] else "NO CUMPLE"],
               ["Aplastamiento bajo la placa: A1 / A1 requerida", f"{m['A1']:.0f} / {m['A1_req']:.0f} cm²",
                "CUMPLE" if m['A1'] >= m['A1_req'] else "NO CUMPLE"],
               ["Altura bajo el borde de la placa (≥ 0.5d)", f"{m['h_ext']:.1f} cm",
                "CUMPLE" if m['h_ext'] >= 0.5 * D.d else "NO CUMPLE"],
               ["Acero del tensor: E.060 / puntal-tensor", f"{m['Asc']:.2f} / {m['As_stm']:.2f} cm²", "se usa el mayor"],
               ["Acero del tensor colocado", f"{m['As_fin']:.2f} cm²", f"{m['nb']} Ø{D.barra_ppal}"],
               ["Anclaje con gancho en la columna, ldh", f"{m['ldh']:.0f} cm",
                "CUMPLE" if m['ldh'] <= D.col_b - rec else "NO CUMPLE"],
               ["Estribos horizontales, Ah", f"{m['Ah']:.2f} cm²", f"{m['ne']} Ø{D.estribo} @ {m['se']:g} cm"]]
    else:
     res = [["Concepto", "Valor", "Verificación"],
           ["Cortante último en la sección crítica, Vu", f"{m.get('Vud', 0):.2f} t", "—"],
           ["Resistencia del concreto, φVc", f"{m.get('phiVc', 0):.2f} t", "—"],
           ["Aporte requerido del acero, Vs", f"{m.get('Vs', 0):.2f} t", "Vs ≤ 2.1√f′c·bw·d: CUMPLE"],
           ["Espaciamiento por cálculo, s", "—" if not m.get("s") or m["s"] > 900 else f"{m['s']:.2f} cm", "—"],
           ["Espaciamiento máximo, s máx", f"{m.get('smax', 0):g} cm", "E.060 11.5.5"]]
    if D.modo not in (4, 5) and R.Xc:
        res += [["Longitud con estribos por cálculo, Xc", f"{R.Xc / 100:.2f} m", "desde la cara"],
                ["Longitud con estribo mínimo, Xm", f"{R.Xm / 100:.2f} m", "desde la cara"]]
    if R.zona_conf:
        res.append(["Zona de confinamiento, 2h", f"{R.zona_conf:g} cm", "E.060 21.4.4"])
    if R.conf_col:
        res.append(["Zona de confinamiento de la columna, Lo", f"{R.conf_col['Lo']:.0f} cm",
                    f"@ {R.conf_col['s0']:g} cm (E.060 21.4.5)"])
    if not M5:
        res.append(["Verificación φVn ≥ Vu en todos los tramos", "Sí" if R.ok else "No", "CUMPLE" if R.ok else "NO CUMPLE"])
    E.append(tabla(res, [8.5, 3.5, 5]))
    for a in R.alertas:
        E.append(Paragraph(f"<font color='#C0392B'><b>Observación:</b> {a}</font>", st_s))

    # ---------------- 6. Detalle de armado ----------------
    E.append(Paragraph(sec("DETALLE DE ARMADO"), st_h1))
    E.append(_fig_img(figura_detalle(D, R, rec), 17))
    E.append(Paragraph("Rojo: tensor de acero Asc. Verde: estribos horizontales Ah. Azul discontinuo: biela de concreto. "
                       "Medidas en cm." if M5 else
                       "Rojo: estribos por cálculo o confinamiento. Azul discontinuo: estribos de montaje "
                       "(zona central donde Vu ≤ φVc/2). Medidas en cm.", st_s))
    E.append(Paragraph(f"{_n[0]}.1 " + ("Cuadro de refuerzo de la ménsula" if M5 else
                                        "Cuadro de estribos (cada extremo, desde la cara del apoyo)"), st_h2))
    cuadro = [["Tramo", "Cantidad", "Espaciamiento", "Desde (cm)", "Hasta (cm)"]]
    if M5:
        cuadro = [["Refuerzo", "Cantidad", "Diámetro", "Ubicación", "Función"],
                  ["Tensor Asc", f"{m['nb']}", D.barra_ppal, "Cara superior", "Equilibra la biela"],
                  ["Estribos Ah", f"{m['ne']}", D.estribo, f"2/3 d superiores @ {m['se']:g} cm", "Cosen la biela"]]
    elif D.modo == 4 and R.conf_col:
        cc = R.conf_col
        x_end = 5 + cc["n0"] * cc["s0"]
        cuadro += [["Primer estribo", "1", "—", "0", "5"],
                   ["Zona de confinamiento Lo", f"{cc['n0']}", f"{cc['s0']:g} cm", "5", f"{x_end:g}"],
                   ["Zona central (resto)", "—", f"{cc['sc']:g} cm", f"{x_end:g}", f"{D.ln_col * 100 - x_end:g}"]]
    elif D.modo == 4:
        cuadro.append(["Todo el elemento", "—", f"{R.tramos[0][1]:g} cm", "—", "—"])
    else:
        cuadro.append(["Primer estribo", "1", "—", "0", f"{R.primero:g}"])
        x, i = R.primero, 1
        half = longitud_viga(D) / 2
        for n, s in R.tramos:
            if n == "resto":
                cnt = int((half - x) // s)
                cuadro.append([f"Tramo {i} (resto)", f"{cnt}", f"{s:g} cm", f"{x:g}", f"{x + cnt * s:g}"])
                x += cnt * s
                break
            cuadro.append([f"Tramo {i}", f"{n}", f"{s:g} cm", f"{x:g}", f"{x + n * s:g}"])
            x += n * s
            i += 1
        pos, _ = posiciones_estribos(D, R)
        nm = sum(1 for _, t in pos if t == "montaje")
        if nm:
            cuadro.append(["Montaje (centro)", f"{nm}", f"{R.smax:g} cm", f"{x:g}", f"{x + nm * R.smax:g}"])
    E.append(tabla(cuadro, [4.5, 2.5, 3.3, 3.3, 3.4]))

    # ---------------- 7. Metrado ----------------
    M = metrado(D, R, rec)
    E.append(Paragraph(sec("METRADO DEL ACERO TRANSVERSAL"), st_h1))
    hh = (D.lm_proy + D.col_b - rec) if M5 else D.h
    E.append(Paragraph(
        f"Longitud de corte de un estribo cerrado: L = 2(bw − 2r) + 2({'l' if M5 else 'h'} − 2r) + 2·g, con gancho a 135° " +
        (f"g = máx(8db; 7.5 cm) = {M['g']:.1f} cm (gancho sísmico)" if gancho(D)[2] else f"g = 6db = {M['g']:.1f} cm") + (f"; longitud del estribo horizontal = lm + bcol − r = {hh:g} cm (ménsula y tramo dentro de la columna)." if M5 else "."), st_s))
    met = [["Descripción", "Valor"],
           ["Longitud de corte por estribo",
            f"2({D.bw:g} − {2 * rec:g}) + 2({hh:g} − {2 * rec:g}) + 2({M['g']:.1f}) = <b>{M['Lc']:.1f} cm</b>"],
           [f"Número de estribos ({M['por']})", f"<b>{M['n']}</b>" + (f"  (cálculo: {M['n_calc']}, montaje: {M['n_mont']})" if M['n_mont'] else "")],
           ["Longitud total de acero", f"{M['L']:.2f} m"],
           [f"Peso unitario Ø{D.estribo}", f"{M['peso']:.3f} kg/m"],
           ["Peso total de estribos", f"<b>{M['kg']:.2f} kg</b>"]]
    E.append(tabla(met, [7, 10]))
    E.append(Paragraph("No incluye desperdicio. Para el pedido de material se recomienda considerar entre 3 % y 5 % adicional.", st_s))

    # ---------------- 8. Especificaciones ----------------
    E.append(Paragraph(sec("ESPECIFICACIONES TÉCNICAS PARA OBRA"), st_h1))
    acero = ("Acero corrugado ASTM A615 Grado 60, fy = 4200 kg/cm²." if D.fy >= 4200 else
             f"Acero de refuerzo con fy = {D.fy:g} kg/cm² (según datos del diseño).")
    notas = [
        f"Concreto: f′c = {D.fc:g} kg/cm² a los 28 días" + (" (concreto liviano, λ = %g)." % D.lam if D.lam < 1 else "."),
        acero,
        f"Recubrimiento libre medido al estribo: {rec:g} cm.",
        f"Estribos Ø{D.estribo} cerrados, con ganchos a 135° y extensión de {M['g']:.1f} cm " + ("(gancho sísmico: 8db, mínimo 7.5 cm). " if gancho(D)[2] else "(6db). ") +
        "Alternar la posición de los ganchos en las esquinas.",
        "Distribución simétrica, medida desde la cara de cada apoyo." if D.modo != 4 else
        "Espaciamiento constante en toda la altura o longitud del elemento.",
    ]
    if M5:
        notas = notas[:3] + [
            f"Tensor: {m['nb']} barras Ø{D.barra_ppal} en la cara superior, ancladas en la columna con gancho "
            f"estándar de 90° (ldh = {m['ldh']:.0f} cm medido desde la cara) y soldadas en el extremo de la ménsula a "
            "una barra transversal del mismo diámetro.",
            f"Estribos horizontales cerrados Ø{D.estribo} ({m['ne']} unidades @ {m['se']:g} cm) en los 2/3 d "
            "superiores, paralelos al tensor.",
            f"Placa de apoyo de {D.placa_l:g} × {D.placa_b:g} cm; no debe sobresalir de la barra transversal del extremo.",
            f"Altura de la ménsula: {D.h:g} cm en la cara de la columna y {D.h_borde:g} cm en el borde libre."]
    elif D.modo != 4:
        notas.append(f"El primer estribo se coloca a {R.primero:g} cm de la cara del apoyo.")
    if R.zona_conf:
        notas.append(f"En las zonas de confinamiento (2h = {R.zona_conf:g} cm desde cada cara) no se permiten "
                     "empalmes por traslape del acero longitudinal.")
    notas += ["Los estribos deben amarrarse firmemente al acero longitudinal con alambre N.° 16 para que no se "
              "desplacen durante el vaciado.",
              "Antes del vaciado, la supervisión verificará número, diámetro, espaciamiento, recubrimiento y "
              "ganchos de los estribos según esta memoria.",
              "Cualquier cambio de sección, carga o material obliga a recalcular el refuerzo."]
    for i, n in enumerate(notas, 1):
        E.append(Paragraph(f"{i}. {n}", st_s))

    # ---------------- 9. Gráficos ----------------
    E.append(PageBreak())
    E.append(Paragraph(sec("GRÁFICOS DE DISEÑO"), st_h1))
    import matplotlib.pyplot as _plt
    for i, (tit, desc, f_) in enumerate(figuras(D, R, rec)[:-1], 1):
        E.append(KeepTogether([Paragraph(f"Gráfico {i}. {tit}", st_h2), _fig_img(f_, 15.5),
                               Paragraph(desc, st_s), Spacer(1, 6)]))
        _plt.close(f_)

    # ---------------- Conclusión ----------------
    if not M5:
        E.append(Paragraph(sec("CONCLUSIÓN"), st_h1))
    if M5:
        concl = (f"La ménsula {elemento} resiste Vu = {D.Vu_sec:g} t mediante una biela de concreto (C = {m['C']:.2f} t) "
                 f"y un tensor de acero (T = {m['T']:.2f} t). Refuerzo: <b>{R.distribucion}</b>. ")
        concl += ("Cumplen la geometría, el aplastamiento bajo la placa, los límites de corte de la E.060, el puntal y "
                  "el nodo del modelo puntal-tensor y el anclaje del tensor." if R.ok
                  else "Revisar las observaciones indicadas.")
        for a in R.alertas:
            concl += f" {a}"
        E.append(Paragraph(sec("CONCLUSIÓN"), st_h1))
        E.append(Paragraph(concl, st_n))
    concl = (f"Para el elemento {elemento}, con Vu = {m.get('Vud', 0):.2f} t en la sección crítica y "
             f"φVc = {m.get('phiVc', 0):.2f} t, se requiere el siguiente refuerzo transversal: "
             f"<b>{R.distribucion}</b>. ")
    concl += ("La sección cumple la resistencia (φVn ≥ Vu), el límite Vs ≤ 2.1√f′c·bw·d, el espaciamiento máximo "
              "y el acero mínimo de la Norma E.060." if R.ok else "Revisar las observaciones indicadas.")
    for a in R.alertas:
        concl += f" {a}"
    if not M5:
        E.append(Paragraph(concl, st_n))

    # ---------------- Referencias ----------------
    E.append(Paragraph(sec("REFERENCIAS"), st_h1))
    for ref in refs:
        E.append(Paragraph(ref.replace("*", ""), st_s))

    # ---------------- Firmas ----------------
    E.append(Spacer(1, 30))
    firmas = Table([["_" * 32, "_" * 32],
                    [Paragraph(f"<b>Elaborado por</b><br/>{info.get('elaborado') or ''}", ParagraphStyle('f', parent=st_s, alignment=TA_CENTER)),
                     Paragraph(f"<b>Revisado por</b><br/>{info.get('revisado') or ''}", ParagraphStyle('f', parent=st_s, alignment=TA_CENTER))]],
                   colWidths=[8.5 * cm, 8.5 * cm])
    firmas.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    E.append(KeepTogether(firmas))

    # ---------------- Plano de armado (lámina A3 girada) ----------------
    try:
        from PIL import Image as PILImage
        from reportlab.platypus import Image as RLImage
        fp = plano_lamina(D, R, rec, info)
        b0 = io.BytesIO()
        fp.savefig(b0, format="png", dpi=150)
        import matplotlib.pyplot as _plt
        _plt.close(fp)
        b0.seek(0)
        im = PILImage.open(b0).rotate(90, expand=True)
        b1 = io.BytesIO()
        im.save(b1, format="PNG")
        b1.seek(0)
        E.append(PageBreak())
        E.append(Paragraph(sec("PLANO DE ARMADO (LÁMINA E-01)"), st_h1))
        E.append(RLImage(b1, width=16.3 * cm, height=16.3 * cm * im.size[1] / im.size[0]))
    except Exception:
        pass

    doc.build(E, onFirstPage=pie, onLaterPages=pie)
    return buf.getvalue()


def _fig_verificaciones(R, titulo):
    """Barras demanda/capacidad de todas las verificaciones (≤ 1 cumple)."""
    import matplotlib.pyplot as plt
    ver = R.ver
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(ver) + 1.4))
    nom = list(ver)[::-1]
    rat = [ver[k][0] / ver[k][1] if ver[k][1] else 0 for k in nom]
    bars = ax.barh(nom, rat, color=["#2E8B62" if r_ <= 1 + 1e-6 else "#C0392B" for r_ in rat])
    ax.axvline(1, color="#333", lw=1.2, ls="--")
    for r_, v in zip(bars, rat):
        ax.text(v, r_.get_y() + r_.get_height() / 2, f"  {v:.2f}", va="center")
    ax.set_xlim(0, max(1.2, max(rat) * 1.2))
    ax.set_xlabel("Demanda / capacidad (debe ser ≤ 1)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(titulo, fontweight="bold")
    fig.tight_layout()
    return fig


def figuras(D, R, rec=4.0):
    """Lista de gráficos del problema: (título, descripción, figura)."""
    if D.modo == 5:
        return figuras_mensula(D, R, rec)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    dbe, Ab = BARRAS[D.estribo]
    Av = D.Av_manual or D.ramas * Ab
    corte_total = D.modo == 4 and D.Nu < 0 and D.corte_total
    Vc = 0.0 if corte_total else R.phiVc / PHI
    AZ, RO, VE, NA, MO = "#0F2A52", "#C0392B", "#2E8B62", "#B0652A", "#6E4BB0"

    if D.modo != 4:
        Lb = longitud_viga(D)
        x = np.linspace(0, Lb, 600)
        half = Lb / 2
        vu_env = np.array([R.Vu(max(min(xi, Lb - xi), D.d)) for xi in x])  # Vu de diseño: constante hasta d
        # ---- 1. Diagramas V y M
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
        if D.modo == 1:
            V = D.wu * (D.L / 2 - x / 100)
            M = D.wu * (x / 100) * (D.L - x / 100) / 2
            a1.plot(x / 100, V, color=AZ, lw=2.2)
            a1.fill_between(x / 100, V, 0, color=AZ, alpha=0.12)
            a2.plot(x / 100, M, color=MO, lw=2.2)
            a2.fill_between(x / 100, M, 0, color=MO, alpha=0.12)
            a2.annotate(f"Mu máx = {M.max():.2f} t·m", (half / 100, M.max()), textcoords="offset points",
                        xytext=(0, -18), ha="center")
            a1.annotate(f"{V[0]:.2f} t", (0, V[0]), textcoords="offset points", xytext=(8, -4))
            a1.annotate(f"{V[-1]:.2f} t", (Lb / 100, V[-1]), textcoords="offset points", xytext=(-48, 4))
            a2.invert_yaxis()
            sub = "Diagramas de cortante y momento (carga última)"
        elif D.modo == 2:
            wc = 1.25 * (D.CM + D.CV)
            a1_ = D.As_sup * D.fy / (0.85 * D.fc * D.bw)
            a2_ = D.As_inf * D.fy / (0.85 * D.fc * D.bw)
            M1 = D.As_sup * D.fy * (D.d - a1_ / 2) / 1e5
            M2 = D.As_inf * D.fy * (D.d - a2_ / 2) / 1e5
            V0 = (M1 + M2) / D.ln + wc * D.ln / 2
            xm = x / 100
            V = V0 - wc * xm
            M = -M1 + V0 * xm - wc * xm ** 2 / 2
            a1.plot(xm, V, color=AZ, lw=2.2, label="Sismo hacia la derecha")
            a1.plot(xm, -V[::-1], color=AZ, lw=1.2, ls="--", label="Sismo hacia la izquierda")
            a1.fill_between(xm, V, 0, color=AZ, alpha=0.10)
            a1.legend(fontsize=8, loc="upper right")
            a2.plot(xm, M, color=MO, lw=2.2)
            a2.fill_between(xm, M, 0, color=MO, alpha=0.12)
            a2.annotate(f"−Mn⁻ = −{M1:.2f}", (0, -M1), textcoords="offset points", xytext=(6, -12))
            a2.annotate(f"Mn⁺ = {M2:.2f}", (D.ln, M2), textcoords="offset points", xytext=(-70, 6))
            a2.invert_yaxis()
            sub = "Cortante y momento por capacidad (rótulas en los extremos)"
        else:
            a1.plot(x / 100, [R.Vu(xi) if xi <= half else -R.Vu(Lb - xi) for xi in x], color=AZ, lw=2.2)
            a1.axhline(0, color="k", lw=0.6)
            a2.plot(x / 100, np.zeros_like(x), color="white")
            a2.text(0.5, 0.5, "Momento: no se dispone del diagrama (solo se ingresó Vu)", ha="center",
                    va="center", transform=a2.transAxes, color="#777")
            sub = "Diagrama de cortante"
        a1.axhline(0, color="k", lw=0.6); a2.axhline(0, color="k", lw=0.6)
        a1.set_ylabel("V (t)"); a2.set_ylabel("M (t·m)"); a2.set_xlabel("x (m)")
        a1.grid(alpha=0.3); a2.grid(alpha=0.3)
        a1.set_title(sub, fontweight="bold")
        fig.tight_layout()
        out.append(("Diagramas de fuerza cortante y momento flector",
                    "Muestran dónde el cortante es mayor (cerca de los apoyos) y cómo varía el momento a lo largo de la viga.",
                    fig))

        # ---- 2. Envolvente φVn vs Vu
        fig, ax = plt.subplots(figsize=(10, 4.6))
        pos, _ = posiciones_estribos(D, R)
        pts = [p for p, _ in pos]
        xs_h, pvn_h = [], []
        for i, p in enumerate(pts):
            s_loc = (pts[i + 1] - p) if i + 1 < len(pts) else (pts[i] - pts[i - 1] if i else R.smax)
            s_loc = max(s_loc, 1)
            xs_h += [p, p + s_loc]
            val = PHI * (Vc + Av * D.fy * D.d / s_loc / 1000)
            pvn_h += [val, val]
        xs_h = np.array(xs_h); pvn_h = np.array(pvn_h)
        ax.plot(x / 100, vu_env, color=RO, lw=2.2, label="Vu de diseño (constante hasta d)")
        ax.plot(xs_h / 100, pvn_h, color=AZ, lw=2, drawstyle="steps-post", label="φVn colocado (capacidad)")
        ax.plot((Lb - xs_h) / 100, pvn_h, color=AZ, lw=2, drawstyle="steps-pre")
        ax.axhline(R.phiVc, color=VE, ls="--", label=f"φVc = {R.phiVc:.2f} t")
        ax.fill_between(x / 100, vu_env, 0, color=RO, alpha=0.08)
        ax.set_xlim(0, Lb / 100); ax.set_ylim(0, max(pvn_h.max(), vu_env.max()) * 1.15)
        ax.set_xlabel("x (m)"); ax.set_ylabel("Cortante (t)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title("Resistencia de diseño φVn frente al cortante último Vu", fontweight="bold")
        fig.tight_layout()
        out.append(("Envolvente de resistencia: φVn ≥ Vu en toda la viga",
                    "La línea azul (capacidad con los estribos colocados) debe quedar siempre por encima de la roja (demanda).",
                    fig))

        # ---- 3. Espaciamiento requerido vs colocado
        fig, ax = plt.subplots(figsize=(10, 4.2))
        xx = np.linspace(0, R.x_fin, 400)
        sr = np.array([min(R.s_req(xi), 100) for xi in xx])
        ax.plot(xx / 100, sr, color=NA, lw=2, label="s requerido por cálculo")
        ax.axhline(R.smax, color=VE, ls="--", label=f"s máx = {R.smax:g} cm")
        xs, ys, p = [], [], R.primero
        for n, s in R.tramos:
            fin = R.x_fin if n == "resto" else p + n * s
            xs += [p / 100, fin / 100]; ys += [s, s]; p = fin
            if n == "resto":
                break
        ax.plot(xs, ys, color=AZ, lw=2.8, label="s colocado")
        ax.fill_between(xs, ys, [min(R.s_req(v * 100), 100) for v in xs], color=AZ, alpha=0.06)
        ax.set_xlim(0, R.x_fin / 100); ax.set_ylim(0, max(R.smax * 1.8, max(ys) * 1.3))
        ax.set_xlabel("Distancia desde la cara del apoyo (m)"); ax.set_ylabel("s (cm)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
        ax.set_title("Espaciamiento requerido y colocado (media viga)", fontweight="bold")
        fig.tight_layout()
        out.append(("Espaciamiento de estribos: requerido y colocado",
                    "El espaciamiento colocado (azul) siempre queda por debajo del requerido y del máximo de la norma.",
                    fig))

        # ---- 4. Aporte por tramo
        fig, ax = plt.subplots(figsize=(10, 4.2))
        etiquetas, vcs, vss, vus = [], [], [], []
        p = R.primero
        for n, s in R.tramos:
            etiquetas.append(f"resto @ {s:g}" if n == "resto" else f"{n} @ {s:g}")
            vcs.append(PHI * Vc); vss.append(PHI * Av * D.fy * D.d / s / 1000); vus.append(R.Vu(max(p, D.d)))
            if n == "resto":
                break
            p += n * s
        idx = np.arange(len(etiquetas))
        ax.bar(idx, vcs, color=VE, label="φVc (concreto)")
        ax.bar(idx, vss, bottom=vcs, color=AZ, label="φVs (estribos)")
        ax.plot(idx, vus, "o", color=RO, ms=10, label="Vu al inicio del tramo")
        for i in idx:
            ax.text(i, vcs[i] + vss[i], f" {vcs[i] + vss[i]:.2f} t", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(idx); ax.set_xticklabels(etiquetas)
        ax.set_ylabel("Cortante (t)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
        ax.set_title("Aporte del concreto y de los estribos en cada tramo", fontweight="bold")
        fig.tight_layout()
        out.append(("Aporte del concreto y del acero por tramo",
                    "Cada barra es la resistencia φVn = φVc + φVs del tramo; el punto rojo es el cortante que debe resistir.",
                    fig))
    else:
        # ---- 1. Barras de resistencia
        out.append(("Resistencia de la sección", "Comparación entre el cortante último y la resistencia con el espaciamiento elegido.",
                    figura(D, R)))
        # ---- 2. φVc vs Nu
        fig, ax = plt.subplots(figsize=(10, 4.3))
        Ag = D.bw * D.h
        nus = np.linspace(-35 * Ag / 1000 * 0.999, max(3 * abs(D.Nu), 140 * Ag / 1000 * 0.6), 300)
        fac = np.where(nus >= 0, 1 + nus * 1000 / (140 * Ag), np.maximum(0, 1 + nus * 1000 / (35 * Ag)))
        pvc = PHI * 0.53 * fac * D.lam * math.sqrt(D.fc) * D.bw * D.d / 1000
        ax.plot(nus, pvc, color=VE, lw=2.2)
        ax.axvline(0, color="k", lw=0.6)
        ax.plot([D.Nu], [R.phiVc], "o", color=RO, ms=10)
        ax.annotate(f"Nu = {D.Nu:g} t\nφVc = {R.phiVc:.2f} t", (D.Nu, R.phiVc), textcoords="offset points",
                    xytext=(10, -30))
        ax.axhline(R.Vu(0), color=RO, ls="--", lw=1, label=f"Vu = {R.Vu(0):.2f} t")
        ax.text(nus.min(), pvc.max() * 0.92, "  TRACCIÓN", color="#555")
        ax.text(nus.max(), pvc.max() * 0.92, "COMPRESIÓN  ", color="#555", ha="right")
        ax.set_xlabel("Carga axial Nu (t)"); ax.set_ylabel("φVc (t)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title("Influencia de la carga axial en el aporte del concreto", fontweight="bold")
        fig.tight_layout()
        out.append(("Efecto de la carga axial en φVc",
                    "La compresión aumenta la resistencia del concreto al corte y la tracción la reduce hasta anularla.",
                    fig))
        # ---- 3. s vs Vu
        fig, ax = plt.subplots(figsize=(10, 4.2))
        vmax = PHI * (Vc + 2.1 * math.sqrt(D.fc) * D.bw * D.d / 1000)
        vv = np.linspace(max(PHI * Vc * 1.01, 0.1), vmax, 300)
        ss = Av * D.fy * D.d / ((vv / PHI - Vc) * 1000)
        ax.plot(vv, np.minimum(ss, 100), color=NA, lw=2, label="s requerido")
        ax.axhline(R.smax, color=VE, ls="--", label=f"s máx = {R.smax:g} cm")
        ax.plot([R.Vu(0)], [R.tramos[0][1]], "o", color=RO, ms=10, label=f"Diseño: s = {R.tramos[0][1]:g} cm")
        ax.set_xlabel("Vu (t)"); ax.set_ylabel("s (cm)"); ax.set_ylim(0, max(60, R.smax * 2)); ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title("Espaciamiento requerido según el cortante último", fontweight="bold")
        fig.tight_layout()
        out.append(("Espaciamiento requerido en función de Vu",
                    "Muestra cuánto debe cerrarse el espaciamiento si el cortante aumenta; arriba de s máx manda la norma.",
                    fig))

    if R.ver:
        out.append(("Resumen de verificaciones (demanda / capacidad)",
                    "Cada barra compara lo que se pide con lo que permite la norma o resiste la sección: "
                    "verde cumple (≤ 1), rojo no cumple.", _fig_verificaciones(R, "Resumen de verificaciones")))
    out.append(("Detalle de armado: elevación y sección",
                "Distribución de estribos para obra y sección con el estribo cerrado, gancho a 135° y recubrimiento.",
                figura_detalle(D, R, rec)))
    return out


# ===========================================================================
# DIBUJOS DE DATOS (SVG) - se actualizan con los valores ingresados
# ===========================================================================
def _flecha(x1, y1, x2, y2, color="#0F2A52", w=2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" '
            f'marker-end="url(#fl)"/>')


def _cota(x1, x2, y, txt, color="#333"):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.2" '
            f'marker-start="url(#fl2)" marker-end="url(#fl)"/>'
            f'<line x1="{x1}" y1="{y - 8}" x2="{x1}" y2="{y + 8}" stroke="{color}"/>'
            f'<line x1="{x2}" y1="{y - 8}" x2="{x2}" y2="{y + 8}" stroke="{color}"/>'
            f'<text x="{(x1 + x2) / 2}" y="{y - 8}" text-anchor="middle" class="t">{txt}</text>')


def _svg(contenido, w, h):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family:Source Sans Pro,Arial,sans-serif">'
            '<defs><marker id="fl" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">'
            '<path d="M0,1 L9,5 L0,9 z" fill="context-stroke"/></marker>'
            '<marker id="fl2" markerWidth="10" markerHeight="10" refX="1" refY="5" orient="auto">'
            '<path d="M9,1 L0,5 L9,9 z" fill="context-stroke"/></marker></defs>'
            '<style>.t{font-size:15px;fill:#1E2A3A}.v{font-size:15px;font-weight:700;fill:#C0392B}'
            '.s{font-size:12px;fill:#6A7890}</style>' + contenido + '</svg>')


def svg_elevacion(D):
    if D.modo == 5:
        return svg_mensula(D)
    c = []
    if D.modo == 1:
        c.append('<rect x="70" y="120" width="500" height="34" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
        c.append('<polygon points="70,154 56,180 84,180" fill="#fff" stroke="#1E2A3A" stroke-width="2"/>')
        c.append('<polygon points="570,154 556,174 584,174" fill="#fff" stroke="#1E2A3A" stroke-width="2"/>'
                 '<circle cx="563" cy="179" r="4" fill="#fff" stroke="#1E2A3A"/><circle cx="577" cy="179" r="4" fill="#fff" stroke="#1E2A3A"/>')
        c.append('<line x1="70" y1="72" x2="570" y2="72" stroke="#C0392B" stroke-width="2"/>')
        for x in range(70, 571, 25):
            c.append(_flecha(x, 72, x, 116, "#C0392B", 1.6))
        c.append(f'<text x="320" y="60" text-anchor="middle" class="v">wu = {D.wu:g} t/m</text>')
        c.append(_cota(70, 570, 215, f"L = {D.L:g} m"))
        c.append('<text x="320" y="245" text-anchor="middle" class="s">Viga simplemente apoyada con carga uniforme</text>')
    elif D.modo == 2:
        for x in (40, 560):
            c.append(f'<rect x="{x}" y="40" width="40" height="200" fill="#BFC5CE" stroke="#1E2A3A" stroke-width="2"/>')
        c.append('<rect x="80" y="110" width="480" height="40" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
        c.append('<line x1="80" y1="70" x2="560" y2="70" stroke="#C0392B" stroke-width="2"/>')
        for x in range(80, 561, 24):
            c.append(_flecha(x, 70, x, 106, "#C0392B", 1.5))
        c.append(f'<text x="320" y="58" text-anchor="middle" class="v">CM = {D.CM:g} t/m · CV = {D.CV:g} t/m</text>')
        c.append('<line x1="80" y1="116" x2="200" y2="116" stroke="#0F2A52" stroke-width="3"/>'
                 '<line x1="440" y1="116" x2="560" y2="116" stroke="#0F2A52" stroke-width="3"/>'
                 '<line x1="80" y1="144" x2="560" y2="144" stroke="#0F2A52" stroke-width="3"/>')
        c.append(f'<text x="140" y="138" text-anchor="middle" class="t">As⁻ = {D.As_sup:.2f} cm²</text>')
        c.append(f'<text x="500" y="138" text-anchor="middle" class="t">As⁻ = {D.As_sup:.2f} cm²</text>')
        c.append(f'<text x="320" y="168" text-anchor="middle" class="t">As⁺ = {D.As_inf:.2f} cm²</text>')
        c.append(f'<rect x="80" y="110" width="{min(2 * D.h / (D.ln * 100) * 480, 200):.0f}" height="40" fill="#6E4BB0" opacity="0.18"/>')
        c.append(_cota(80, 560, 210, f"ln = {D.ln:g} m"))
        c.append('<text x="320" y="245" text-anchor="middle" class="s">Viga de pórtico entre columnas (sismo: diseño por capacidad)</text>')
    elif D.modo == 3:
        for x in (40, 564):
            c.append(f'<rect x="{x}" y="22" width="36" height="110" fill="#BFC5CE" stroke="#1E2A3A" stroke-width="2"/>')
        c.append('<rect x="76" y="50" width="488" height="32" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
        # diagrama de cortante: +Vu en la cara izquierda, 0 al centro, −Vu en la cara derecha
        c.append('<line x1="76" y1="150" x2="564" y2="150" stroke="#333"/>')
        c.append('<polygon points="76,150 76,110 320,150" fill="#C0392B" opacity="0.25" stroke="#C0392B" stroke-width="2"/>')
        c.append('<polygon points="320,150 564,190 564,150" fill="#C0392B" opacity="0.25" stroke="#C0392B" stroke-width="2"/>')
        c.append(f'<text x="84" y="104" class="v">+Vu = {D.Vu_cara:g} t</text>')
        c.append(f'<text x="556" y="206" text-anchor="end" class="v">−Vu = {D.Vu_cara:g} t</text>')
        c.append('<line x1="320" y1="40" x2="320" y2="200" stroke="#777" stroke-dasharray="6 4"/>')
        c.append('<text x="326" y="146" class="t">Vu = 0</text>')
        c.append(_cota(76, 320, 222, f"{D.luz_mitad:g} m (cara → Vu = 0)"))
        c.append(_cota(76, 564, 250, f"luz libre = {2 * D.luz_mitad:g} m (simétrica)"))
    elif D.Nu < 0:                           # tracción axial: elemento horizontal (viga en flexotracción)
        c.append('<rect x="110" y="100" width="420" height="70" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
        s = D.s_exist or 20
        n = max(4, min(16, int(420 / max(s, 5) * 0.6)))
        for i in range(n):
            x = 122 + i * 396 / (n - 1)
            c.append(f'<line x1="{x:.0f}" y1="104" x2="{x:.0f}" y2="166" stroke="#C0392B" stroke-width="1.5"/>')
        c.append(_flecha(106, 135, 40, 135, "#0F2A52", 3) + _flecha(534, 135, 600, 135, "#0F2A52", 3))
        c.append(f'<text x="400" y="88" text-anchor="middle" class="v">Nu = {abs(D.Nu):g} t (tracción)</text>')
        c.append(_flecha(200, 30, 200, 96, "#C0392B", 3))
        c.append(f'<text x="210" y="40" class="v">Vu = {D.Vu_sec:g} t</text>')
        if D.s_exist:
            c.append(f'<text x="320" y="196" text-anchor="middle" class="t">Ø{D.estribo} @ {D.s_exist:g} cm (existente)</text>')
        c.append('<text x="320" y="240" text-anchor="middle" class="s">Viga con cortante y tracción axial (flexotracción)</text>')
    else:
        comp = D.Nu >= 0
        c.append('<rect x="280" y="40" width="80" height="200" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
        s = D.s_exist or 20
        n = max(3, min(12, int(200 / max(s, 5) * 1.2)))
        for i in range(n):
            y = 50 + i * 180 / (n - 1)
            c.append(f'<line x1="284" y1="{y:.0f}" x2="356" y2="{y:.0f}" stroke="#C0392B" stroke-width="1.5"/>')
        if comp:
            c.append(_flecha(320, 0, 320, 36, "#0F2A52", 3))
        else:
            c.append(_flecha(320, 38, 320, 2, "#0F2A52", 3))
        c.append(f'<text x="340" y="22" class="v">Nu = {abs(D.Nu):g} t {"(compresión)" if comp else "(tracción)"}</text>')
        c.append(_flecha(180, 140, 276, 140, "#C0392B", 3))
        c.append(f'<text x="150" y="128" class="v">Vu = {D.Vu_sec:g} t</text>')
        if D.s_exist:
            c.append(f'<text x="370" y="150" class="t">Ø{D.estribo} @ {D.s_exist:g} cm (existente)</text>')
        c.append('<text x="320" y="262" text-anchor="middle" class="s">Elemento con cortante y carga axial</text>')
    return _svg("".join(c), 640, 270 if D.modo != 3 else 268)


def svg_seccion_mensula(D):
    """Corte de la ménsula en la cara de la columna: tensor arriba y estribos horizontales Ah."""
    try:
        R = calcular(D)
        nb, ne, se = R.metr["nb"], R.metr["ne"], R.metr["se"]
    except Exception:
        nb, ne, se = 3, 2, 10
    escala = 170 / max(D.h, D.bw * 1.2)
    w, h = D.bw * escala, D.h * escala
    x0, y0 = 70, 30
    r = 4 * escala
    c = [f'<rect x="{x0}" y="{y0}" width="{w:.1f}" height="{h:.1f}" fill="#E5E8EC" stroke="#1E2A3A" stroke-width="2"/>']
    yt = y0 + (D.h - D.d) * escala
    rb = max(4, BARRAS[D.barra_ppal][0] / 2 * escala)
    for i in range(nb):
        cx = x0 + r + rb + (w - 2 * r - 2 * rb) * i / max(nb - 1, 1)
        c.append(f'<circle cx="{cx:.1f}" cy="{yt:.1f}" r="{rb:.1f}" fill="#C0392B"/>')
    for i in range(ne):
        yy = yt + (i + 1) * se * escala
        c.append(f'<line x1="{x0 + r:.1f}" y1="{yy:.1f}" x2="{x0 + w - r:.1f}" y2="{yy:.1f}" stroke="#2E8B62" stroke-width="3"/>')
        for cx in (x0 + r, x0 + w - r):
            c.append(f'<circle cx="{cx:.1f}" cy="{yy:.1f}" r="3.5" fill="#2E8B62"/>')
    c.append(_cota(x0, x0 + w, y0 + h + 22, f"bw = {D.bw:g}"))
    c.append(f'<line x1="{x0 - 22}" y1="{y0}" x2="{x0 - 22}" y2="{y0 + h:.1f}" stroke="#333" '
             f'marker-start="url(#fl2)" marker-end="url(#fl)"/>'
             f'<text x="{x0 - 28}" y="{y0 + h / 2:.1f}" text-anchor="end" class="t">h = {D.h:g}</text>')
    c.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 52:.1f}" text-anchor="middle" class="v">'
             f'Tensor {nb} Ø{D.barra_ppal} + Ah {ne} □Ø{D.estribo}</text>')
    c.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 70:.1f}" text-anchor="middle" class="s">'
             f'corte en la cara de la columna · sin estribos verticales</text>')
    return _svg("".join(c), 300, 30 + h + 80)


def svg_seccion(D):
    escala = 170 / max(D.h, D.bw * 1.2)
    w, h = D.bw * escala, D.h * escala
    x0, y0 = 70, 30
    r = 4 * escala
    c = [f'<rect x="{x0}" y="{y0}" width="{w:.1f}" height="{h:.1f}" fill="#E5E8EC" stroke="#1E2A3A" stroke-width="2"/>',
         f'<rect x="{x0 + r:.1f}" y="{y0 + r:.1f}" width="{w - 2 * r:.1f}" height="{h - 2 * r:.1f}" fill="none" '
         f'stroke="#C0392B" stroke-width="3" rx="3"/>',
         f'<line x1="{x0 + r:.1f}" y1="{y0 + r:.1f}" x2="{x0 + r + 14:.1f}" y2="{y0 + r + 14:.1f}" stroke="#C0392B" stroke-width="3"/>']
    rb = max(4, 0.95 * escala)
    if D.modo == 5:
        return svg_seccion_mensula(D)
    xs_ = [x0 + r + rb + 2, x0 + w - r - rb - 2]
    ys_ = [y0 + r + rb + 2, y0 + h - r - rb - 2]
    for cx in xs_:
        for cy in ys_:
            c.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rb:.1f}" fill="#1E2A3A"/>')
    if D.modo == 4 and max(D.bw, D.h) > 45:          # barras intermedias, igual que en el plano y el 3D
        for cx in ((xs_[0] + xs_[1]) / 2,):
            for cy in ys_:
                c.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rb:.1f}" fill="#1E2A3A"/>')
        for cy in ((ys_[0] + ys_[1]) / 2,):
            for cx in xs_:
                c.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rb:.1f}" fill="#1E2A3A"/>')
    yd = y0 + D.d * escala
    c.append(_cota(x0, x0 + w, y0 + h + 22, f"bw = {D.bw:g}"))
    c.append(f'<line x1="{x0 - 22}" y1="{y0}" x2="{x0 - 22}" y2="{y0 + h:.1f}" stroke="#333" '
             f'marker-start="url(#fl2)" marker-end="url(#fl)"/>'
             f'<text x="{x0 - 28}" y="{y0 + h / 2:.1f}" text-anchor="end" class="t">h = {D.h:g}</text>')
    c.append(f'<line x1="{x0 + w + 16:.1f}" y1="{y0}" x2="{x0 + w + 16:.1f}" y2="{yd:.1f}" stroke="#2E8B62" '
             f'marker-start="url(#fl2)" marker-end="url(#fl)"/>'
             f'<text x="{x0 + w + 22:.1f}" y="{(y0 + yd) / 2:.1f}" class="t" fill="#2E8B62">d = {D.d:g}</text>')
    c.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 52:.1f}" text-anchor="middle" class="v">'
             f'Estribo Ø{D.estribo} · {D.ramas} ramas</text>')
    c.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 70:.1f}" text-anchor="middle" class="s">medidas en cm</text>')
    return _svg("".join(c), 300, 30 + h + 80)



# ===========================================================================
# MODO 5: MÉNSULA CORTA - el corte se resiste con una BIELA (puntal) de concreto
# y un TENSOR de acero, no con estribos (Perles, 2009, p. 192; E.060 ménsulas)
# ===========================================================================
REF_PERLES = "Perles, P. (2009). *Hormigón armado* (Tomo I). Nobuko."
REF_ACI = "American Concrete Institute [ACI]. (2008). *Requisitos de reglamento para concreto estructural (ACI 318S-08)*, Apéndice A: modelos puntal-tensor."


def calcular_mensula(D: Datos) -> Resultado:
    fk = lambda v: f"{v:,.0f}".replace(",", "\\,")
    R = Resultado()
    R.modo = 5
    rec = 4.0
    sq = math.sqrt(D.fc)
    bw, h, d, a = D.bw, D.h, D.d, D.a_carga
    lp, bp, lm, hb = D.placa_l, D.placa_b, D.lm_proy, D.h_borde
    Vu = D.Vu_sec * 1000            # kg
    db_p, Ab_p = BARRAS[D.barra_ppal]
    dbe, Ab_e = BARRAS[D.estribo]
    mu = 1.4 * D.lam
    bp_ = D.barra_ppal.replace(chr(34), "")
    R.texto.append(f"MÉNSULA CORTA: {D.titulo}")
    chk = lambda ok: r"\;\checkmark" if ok else r"\;\times"
    ver = {}                         # verificaciones: nombre -> (demanda, capacidad)

    # 1. Geometría
    rel = a / d
    Nuc_min = 0.2 * D.Vu_sec
    Nuc = max(D.Nuc, Nuc_min)
    x_ext = a + lp / 2                                   # borde exterior de la placa
    h_ext = h - (h - hb) * min(x_ext / lm, 1.0)          # altura bajo el borde exterior de la placa
    x_lim = lm - rec - db_p                              # cara interior de la barra transversal del extremo
    ok_ad, ok_nuc = rel <= 1, Nuc <= D.Vu_sec + 1e-9
    ok_hb, ok_pl = h_ext >= 0.5 * d - 1e-9, x_ext <= x_lim + 1e-9
    R.paso("1. Geometría y condición de ménsula corta",
           [rf"\frac{{a}}{{d}} = \frac{{{a:g}}}{{{d:g}}} = {rel:.2f} " + (r"\le 1" if ok_ad else r"> 1") + chk(ok_ad),
            rf"0.2\,V_u = {Nuc_min:.2f} \le N_{{uc}} = {Nuc:.2f} " + (r"\le" if ok_nuc else ">") + rf" V_u = {D.Vu_sec:g}\ \text{{t}}" + chk(ok_nuc),
            rf"h_{{ext}} = h - (h - h_b)\frac{{a + l_p/2}}{{l_m}} = {h:g} - ({h:g} - {hb:g})\frac{{{x_ext:g}}}{{{lm:g}}} = {h_ext:.1f}\ \text{{cm}} "
            + (r"\ge" if ok_hb else "<") + rf" 0.5\,d = {0.5 * d:.1f}\ \text{{cm}}" + chk(ok_hb),
            rf"a + \frac{{l_p}}{{2}} = {x_ext:g}\ \text{{cm}} " + (r"\le" if ok_pl else ">")
            + rf" l_m - r - d_b = {lm:g} - {rec:g} - {db_p:.2f} = {x_lim:.1f}\ \text{{cm}}" + chk(ok_pl)],
           "Es ménsula corta si la carga actúa a una distancia no mayor que d. La fuerza horizontal Nuc (retracción, "
           "temperatura, fricción del apoyo) se toma como mínimo 0.2·Vu y no debe superar Vu. La altura bajo el borde exterior de la placa "
           "no debe ser menor que 0.5d, y la placa no debe pasar de la barra transversal donde se ancla el tensor; "
           "si no, la esquina se desprende.")
    if not ok_nuc:
        R.alertas.append("Nuc > Vu: fuera del alcance de las disposiciones de ménsulas de la E.060.")
    if not ok_ad:
        R.alertas.append("a/d > 1: no es ménsula corta; debe diseñarse como voladizo (flexión y corte con estribos).")
    if not ok_hb:
        R.alertas.append(f"La altura bajo el borde exterior de la placa ({h_ext:.1f} cm) es menor que 0.5d: aumente h_b.")
    if not ok_pl:
        R.alertas.append("La placa de apoyo sobrepasa la barra transversal del extremo: aumente la proyección lm o reduzca a.")

    # 2. Aplastamiento bajo la placa
    A1 = lp * bp
    phib = 0.70
    Pb = phib * 0.85 * D.fc * A1
    A1_req = Vu / (phib * 0.85 * D.fc)
    ok_ap = Vu <= Pb
    ver["Aplastamiento bajo la placa"] = (Vu, Pb)
    R.paso("2. Aplastamiento del concreto bajo la placa de apoyo",
           [rf"A_1 = l_p\,b_p = {lp:g}\times{bp:g} = {A1:g}\ \text{{cm}}^2",
            rf"\phi P_{{nb}} = \phi\,0.85 f'_c A_1 = 0.70\times0.85\times{D.fc:g}\times{A1:g} = {fk(Pb)}\ \text{{kg}}",
            rf"V_u = {fk(Vu)}\ " + (r"\le" if ok_ap else ">") + rf"\ {fk(Pb)}\ \text{{kg}}" + chk(ok_ap)
            + rf";\quad A_{{1,req}} = \frac{{V_u}}{{0.70\times0.85 f'_c}} = {A1_req:.0f}\ \text{{cm}}^2"],
           "La carga entra a la ménsula por una placa de acero. El concreto debajo no debe aplastarse "
           "(φ = 0.70 para aplastamiento). Si no cumple, se agranda la placa.")
    if not ok_ap:
        R.alertas.append(f"Aplastamiento bajo la placa: se necesita A1 ≥ {A1_req:.0f} cm².")

    # 3. Modelo biela - tensor simplificado (Perles)
    z = 0.9 * d
    th = math.atan(z / a)
    T = D.Vu_sec * a / z + Nuc
    C = D.Vu_sec / math.sin(th)
    As_t = T * 1000 / (PHI * D.fy)
    R.biela = dict(z=z, th=th, T=T, C=C, As_t=As_t, Nuc=Nuc)
    R.paso("3. Modelo simplificado de biela y tensor",
           [rf"z \approx 0.9\,d = 0.9\times{d:g} = {z:.1f}\ \text{{cm}};\qquad \tan\theta = \frac{{z}}{{a}} = \frac{{{z:.1f}}}{{{a:g}}} \Rightarrow \theta = {math.degrees(th):.1f}^\circ",
            rf"T = V_u\frac{{a}}{{z}} + N_{{uc}} = {D.Vu_sec:g}\times\frac{{{a:g}}}{{{z:.1f}}} + {Nuc:.2f} = {T:.2f}\ \text{{t}}\quad\text{{(tensor)}}",
            rf"C = \frac{{V_u}}{{\sin\theta}} = \frac{{{D.Vu_sec:g}}}{{\sin {math.degrees(th):.1f}^\circ}} = {C:.2f}\ \text{{t}}\quad\text{{(biela de concreto)}}",
            rf"A_{{s}} = \frac{{T}}{{\phi f_y}} = \frac{{{fk(T * 1000)}}}{{0.85\times{D.fy:g}}} = {As_t:.2f}\ \text{{cm}}^2"],
           "La carga baja al apoyo por una biela inclinada de concreto en compresión y se equilibra con un tensor "
           "horizontal de acero en la parte superior (triángulo de fuerzas). Los estribos no son el refuerzo "
           "principal: el corte lo toma la biela.")

    # 4. Resistencia máxima al corte
    Vn = Vu / PHI
    if D.lam >= 1:      # E.060 11.9.3.2.1 (concreto de peso normal): 0.2 f'c bw d y 5.5 MPa·bw·d = 56 kg/cm²
        l1, l2 = 0.2 * D.fc * bw * d, 56 * bw * d
        t1, t2 = r"0.2 f'_c\,b_w d", r"56\,b_w d"
    else:               # E.060 11.9.3.2.2 (concreto liviano)
        l1, l2 = (0.2 - 0.07 * rel) * D.fc * bw * d, (56 - 19.4 * rel) * bw * d
        t1, t2 = rf"(0.2 - 0.07\,a/d) f'_c\,b_w d", rf"(56 - 19.4\,a/d)\,b_w d"
    ok_v = Vn <= min(l1, l2)
    ver["Vn ≤ 0.2·f′c·bw·d"] = (Vn, l1)
    ver["Vn ≤ 56·bw·d"] = (Vn, l2)
    R.paso("4. Resistencia máxima al corte de la ménsula",
           [rf"V_n = \frac{{V_u}}{{\phi}} = \frac{{{fk(Vu)}}}{{0.85}} = {fk(Vn)}\ \text{{kg}}",
            rf"V_n \le {t1} = {fk(l1)}\ \text{{kg}}" + chk(Vn <= l1),
            rf"V_n \le {t2} = {fk(l2)}\ \text{{kg}}" + chk(Vn <= l2)],
           "Estos límites evitan que el concreto de la ménsula se aplaste en la cara de la columna. Si no cumplen, "
           "se aumenta bw, d o f′c.")
    if not ok_v:
        R.alertas.append("Vn supera el límite de corte de la ménsula: aumente las dimensiones o f′c.")

    # 5. Acero por flexión, tracción y corte-fricción
    Mu = D.Vu_sec * a + Nuc * (h - d)          # t·cm
    Af = Mu * 1000 / (PHI * D.fy * 0.9 * d)
    An = Nuc * 1000 / (PHI * D.fy)
    Avf = Vn / (D.fy * mu)
    R.paso("5. Acero para flexión, tracción directa y corte-fricción",
           [rf"M_u = V_u\,a + N_{{uc}}(h-d) = {D.Vu_sec:g}\times{a:g} + {Nuc:.2f}\times({h:g}-{d:g}) = {Mu:.1f}\ \text{{t·cm}}",
            rf"A_f = \frac{{M_u}}{{\phi f_y (0.9 d)}} = \frac{{{fk(Mu * 1000)}}}{{0.85\times{D.fy:g}\times{0.9 * d:.1f}}} = {Af:.2f}\ \text{{cm}}^2",
            rf"A_n = \frac{{N_{{uc}}}}{{\phi f_y}} = \frac{{{fk(Nuc * 1000)}}}{{0.85\times{D.fy:g}}} = {An:.2f}\ \text{{cm}}^2",
            rf"A_{{vf}} = \frac{{V_n}}{{f_y\,\mu}} = \frac{{{fk(Vn)}}}{{{D.fy:g}\times{mu:.2f}}} = {Avf:.2f}\ \text{{cm}}^2;\quad \mu = 1.4\lambda"],
           "Af resiste el momento en la cara de la columna, An la tracción horizontal y Avf el deslizamiento "
           "por corte-fricción en la cara de la columna (concreto vaciado monolíticamente, μ = 1.4λ).")

    # 6. Acero principal (tensor) Asc
    o1, o2 = Af + An, 2 / 3 * Avf + An
    amin = 0.04 * D.fc / D.fy * bw * d
    Asc = max(o1, o2, amin)
    ctrl = {o1: "Af + An", o2: "2/3·Avf + An", amin: "Asc mín"}[Asc]
    R.paso("6. Acero principal del tensor, Asc",
           [rf"A_{{sc}} \ge A_f + A_n = {Af:.2f} + {An:.2f} = {o1:.2f}\ \text{{cm}}^2",
            rf"A_{{sc}} \ge \frac{{2}}{{3}}A_{{vf}} + A_n = \frac{{2}}{{3}}({Avf:.2f}) + {An:.2f} = {o2:.2f}\ \text{{cm}}^2",
            rf"A_{{sc,mín}} = 0.04\frac{{f'_c}}{{f_y}}b_w d = 0.04\times\frac{{{D.fc:g}}}{{{D.fy:g}}}\times{bw:g}\times{d:g} = {amin:.2f}\ \text{{cm}}^2",
            rf"A_{{sc}} = \max({o1:.2f};\ {o2:.2f};\ {amin:.2f}) = {Asc:.2f}\ \text{{cm}}^2\quad\text{{(controla {ctrl})}}"],
           "Asc es el tensor del modelo: barras horizontales en la cara superior, ancladas en la columna y "
           "en el extremo de la ménsula.")

    # 7. Estribos horizontales cerrados Ah
    Ah = max(0.0, 0.5 * (Asc - An))
    Av_e = 2 * Ab_e
    ne = max(2, math.ceil(Ah / Av_e - 1e-9))
    zona = 2 / 3 * d
    se = rd(zona / ne, 1.0)
    R.paso("7. Estribos cerrados horizontales, Ah",
           [rf"A_h = 0.5\,(A_{{sc}} - A_n) = 0.5\,({Asc:.2f} - {An:.2f}) = {Ah:.2f}\ \text{{cm}}^2",
            rf"n = \frac{{A_h}}{{2A_b}} = \frac{{{Ah:.2f}}}{{2\times{Ab_e:.2f}}} \Rightarrow {ne}\ \text{{estribos}}\ \text{{Ø}}{D.estribo.replace(chr(34), '')}''",
            rf"\text{{En}}\ \frac{{2}}{{3}}d = {zona:.1f}\ \text{{cm}}\ \text{{bajo el tensor}} \Rightarrow @\ {se:g}\ \text{{cm}}"],
           "Estos estribos son paralelos a Asc y cosen la biela para que no se raje a lo largo (falla por "
           "hendimiento). No son los estribos verticales de una viga.")

    # 8. Verificación por el método puntal-tensor (ACI 318)
    ps = 0.75
    zz = 0.9 * d
    for _ in range(30):                                  # nodo inferior: bloque comprimido en la cara de la columna
        Cc = D.Vu_sec * a / zz * 1000
        xc = Cc / (ps * 0.85 * 1.0 * D.fc * bw)
        zz = d - xc / 2
    th2 = math.atan(zz / a)
    T2 = D.Vu_sec * a / zz + Nuc
    C2 = D.Vu_sec / math.sin(th2)
    As_stm = T2 * 1000 / (ps * D.fy)
    rho_h = Av_e / (bw * se) * math.sin(th2)
    bs = 0.75 if rho_h >= 0.003 else 0.60 * D.lam
    wt = 2 * (h - d)
    ws = lp * math.sin(th2) + wt * math.cos(th2)
    Fns = ps * 0.85 * bs * D.fc * ws * bw
    Fnn = ps * 0.85 * 0.80 * D.fc * lp * bp
    ok_s, ok_n = C2 * 1000 <= Fns, Vu <= Fnn
    ver["Puntal (biela) en el nodo superior"] = (C2 * 1000, Fns)
    ver["Nodo bajo la carga (CCT)"] = (Vu, Fnn)
    R.stm = dict(z=zz, th=th2, T=T2, C=C2, As=As_stm, xc=xc, ws=ws, bs=bs, rho=rho_h)
    R.paso("8. Verificación por el modelo puntal-tensor (ACI 318-08, Apéndice A; φ = 0.75)",
           [rf"\text{{Nodo inferior:}}\ x = \frac{{V_u\,a/z}}{{\phi\,0.85 f'_c b_w}};\quad z = d - \frac{{x}}{{2}} \Rightarrow x = {xc:.2f}\ \text{{cm}},\ z = {zz:.2f}\ \text{{cm}}",
            rf"\theta = \arctan\frac{{{zz:.2f}}}{{{a:g}}} = {math.degrees(th2):.1f}^\circ;\quad T = {D.Vu_sec:g}\frac{{{a:g}}}{{{zz:.2f}}} + {Nuc:.2f} = {T2:.2f}\ \text{{t}};\quad C = {C2:.2f}\ \text{{t}}",
            rf"A_s = \frac{{T}}{{\phi f_y}} = \frac{{{fk(T2 * 1000)}}}{{0.75\times{D.fy:g}}} = {As_stm:.2f}\ \text{{cm}}^2",
            rf"\rho_h = \frac{{2A_b}}{{b_w s}}\sin\theta = \frac{{{Av_e:.2f}}}{{{bw:g}\times{se:g}}}\sin{math.degrees(th2):.1f}^\circ = {rho_h:.4f} "
            + (r"\ge 0.003 \Rightarrow \beta_s = 0.75" if rho_h >= 0.003 else rf"< 0.003 \Rightarrow \beta_s = 0.60\lambda = {bs:.2f}"),
            rf"w_s = l_p\sin\theta + w_t\cos\theta = {lp:g}\sin{math.degrees(th2):.1f}^\circ + {wt:g}\cos{math.degrees(th2):.1f}^\circ = {ws:.1f}\ \text{{cm}}\quad (w_t = 2(h-d))",
            rf"\phi F_{{ns}} = 0.75\times0.85\,\beta_s f'_c\,w_s b_w = 0.75\times0.85\times{bs:.2f}\times{D.fc:g}\times{ws:.1f}\times{bw:g} = {fk(Fns)}\ \text{{kg}} "
            + (r"\ge" if ok_s else "<") + rf" C = {fk(C2 * 1000)}" + chk(ok_s),
            rf"\text{{Nodo CCT:}}\ \phi F_{{nn}} = 0.75\times0.85\times0.80\times{D.fc:g}\times{lp:g}\times{bp:g} = {fk(Fnn)}\ \text{{kg}} "
            + (r"\ge" if ok_n else "<") + rf" V_u" + chk(ok_n)],
           "Este es el método general para las zonas donde no vale la hipótesis de Bernoulli. Se ubica el bloque "
           "comprimido en la cara de la columna para hallar el brazo z real, se calcula la fuerza del tensor y se "
           "verifica que el puntal y el nodo bajo la carga no se aplasten. βs = 0.75 si los estribos Ah cosen el "
           "puntal (ρ ≥ 0.003); el nodo con un tensor anclado (CCT) usa βn = 0.80.")
    if not ok_s:
        R.alertas.append("El puntal (biela) no resiste en el nodo superior: aumente la placa, bw o f′c.")
    if not ok_n:
        R.alertas.append("El nodo bajo la carga no resiste: agrande la placa de apoyo.")

    # 9. Comparación de métodos y acero final
    As_fin = max(Asc, As_stm)
    nb = max(2, math.ceil(As_fin / Ab_p - 1e-9))
    s_lib = (bw - 2 * rec - 2 * dbe - nb * db_p) / max(nb - 1, 1)
    ok_esp = s_lib >= max(2.5, db_p) - 1e-9
    R.paso("9. Comparación de métodos y acero del tensor a colocar",
           [rf"\text{{Norma (empírico):}}\ A_{{sc}} = {Asc:.2f};\quad \text{{simplificado:}}\ {As_t:.2f};\quad \text{{puntal-tensor:}}\ {As_stm:.2f}\ \text{{cm}}^2",
            rf"A_s = \max({Asc:.2f};\ {As_stm:.2f}) = {As_fin:.2f}\ \text{{cm}}^2 \Rightarrow n = \frac{{{As_fin:.2f}}}{{{Ab_p:.2f}}} \Rightarrow {nb}\ \text{{Ø}}{bp_}''\ ({nb * Ab_p:.2f}\ \text{{cm}}^2)",
            rf"s_{{libre}} = \frac{{b_w - 2r - 2d_{{be}} - n\,d_b}}{{n-1}} = \frac{{{bw:g} - {2 * rec:g} - {2 * dbe:.2f} - {nb}\times{db_p:.2f}}}{{{max(nb - 1, 1)}}} = {s_lib:.1f}\ \text{{cm}} "
            + (r"\ge" if ok_esp else "<") + rf" \max(2.5;\ d_b)" + chk(ok_esp)],
           "Los tres caminos dan valores parecidos, lo que confirma el modelo. Se coloca el mayor para quedar del "
           "lado seguro y se verifica que las barras quepan en el ancho con el espaciamiento libre mínimo.")
    if not ok_esp:
        R.alertas.append("Las barras del tensor no caben en una capa: use un diámetro mayor o aumente bw.")

    # 10. Anclaje del tensor
    fl = 1.0 if D.lam >= 1 else 1.3          # E.060 cap. 12: λ = 1.3 multiplica ld en concreto liviano
    ldh = max(0.075 * fl * D.fy / sq * db_p, 8 * db_p, 15)
    disp = D.col_b - rec
    ok_ldh = ldh <= disp + 1e-9
    k_ld = 6.6 if db_p <= 1.91 else 5.3      # E.060 12.2.2 (equivale a 25 y 20 del ACI en psi)
    ld = max(D.fy * 1.3 * fl / (k_ld * sq) * db_p, 30)
    l_ext = lm - x_ext - rec
    ver["Anclaje con gancho en la columna"] = (ldh, disp)
    R.anc = dict(ldh=ldh, disp=disp, ld=ld, l_ext=l_ext)
    R.paso("10. Anclaje del tensor",
           [rf"\text{{En la columna (gancho a 90}}^\circ\text{{):}}\ l_{{dh}} = \frac{{0.075\,\lambda f_y}}{{\sqrt{{f'_c}}}}d_b = \frac{{0.075\times{fl:g}\times{D.fy:g}}}{{{sq:.2f}}}\times{db_p:.2f} = {0.075 * fl * D.fy / sq * db_p:.1f}\ \text{{cm}}\ (\ge 8d_b;\ 15\ \text{{cm}}) \Rightarrow {ldh:.1f}\ \text{{cm}}",
            rf"\text{{Disponible:}}\ b_{{col}} - r = {D.col_b:g} - {rec:g} = {disp:.1f}\ \text{{cm}}\ " + (r"\ge" if ok_ldh else "<") + rf"\ {ldh:.1f}" + chk(ok_ldh),
            rf"\text{{En el extremo:}}\ l_d = \frac{{f_y\,\psi_t\,\lambda}}{{{k_ld}\sqrt{{f'_c}}}}d_b = \frac{{{D.fy:g}\times1.3\times{fl:g}}}{{{k_ld}\times{sq:.2f}}}\times{db_p:.2f} = {ld:.0f}\ \text{{cm}}\ " + (">" if ld > l_ext else r"\le") + rf"\ {l_ext:.1f}\ \text{{cm}}\ \text{{disponibles}}",
            (r"\Rightarrow \text{soldar el tensor a una barra transversal del mismo diámetro (anclaje mecánico)}"
             if ld > l_ext else r"\Rightarrow \text{anclaje recto suficiente; igual se recomienda la barra transversal}")],
           "El tensor debe desarrollar fy a ambos lados del nodo. Dentro de la columna se ancla con gancho estándar "
           "de 90° (longitud ldh, E.060 cap. 12). Hacia el borde libre no hay longitud para un anclaje recto, por eso "
           "se suelda a una barra transversal o se dobla en lazo horizontal: este es el tema de longitud de desarrollo.")
    if not ok_ldh:
        R.alertas.append(f"El gancho del tensor necesita ldh = {ldh:.0f} cm y la columna solo da {disp:.0f} cm: "
                         "use barras de menor diámetro o una columna más ancha.")

    # 11. Respuesta
    R.tramos = [("mensula", se)]
    R.primero = 0
    R.ver = ver
    R.ok = all(dem <= cap + 1e-6 for dem, cap in ver.values()) and ok_v and ok_ad and ok_hb and ok_pl and ok_esp and ok_nuc
    R.distribucion = (f"Tensor: {nb} Ø{D.barra_ppal} ({nb * Ab_p:.2f} cm²) con gancho de {ldh:.0f} cm en la columna y "
                      f"barra transversal soldada en el extremo + {ne} estribos cerrados Ø{D.estribo} @ {se:g} cm "
                      f"en los 2/3 d superiores")
    R.paso("11. Respuesta",
           [rf"\text{{Tensor:}}\ {nb}\ \text{{Ø}}{bp_}''\ ({nb * Ab_p:.2f}\ \text{{cm}}^2);\quad \text{{Ah:}}\ {ne}\ \text{{estribos Ø}}{D.estribo.replace(chr(34), '')}''\ @\ {se:g}\ \text{{cm}}"],
           ("Todas las verificaciones cumplen. " if R.ok else "Hay verificaciones que no cumplen: ver observaciones. ")
           + "El corte de la ménsula lo resiste la biela de concreto; el acero no son estribos verticales sino un "
             "tensor horizontal bien anclado y estribos horizontales que cosen la biela.")
    R.metr = dict(Vud=D.Vu_sec, phiVc=0.0, Vs=0.0, s=se, smax=se, Asc=Asc, nb=nb, Ah=Ah, ne=ne, se=se,
                  Vn=Vn, lim=min(l1, l2), Af=Af, An=An, Avf=Avf, amin=amin, o1=o1, o2=o2, As_t=As_t, T=T, C=C,
                  As_stm=As_stm, As_fin=As_fin, ldh=ldh, h_ext=h_ext, A1=A1, A1_req=A1_req)
    R.texto.append("\nRESULTADO: " + R.distribucion)
    R.texto.append("\nREFERENCIAS")
    R.texto += ["  " + r.replace("*", "") for r in REFERENCIAS + [REF_PERLES, REF_ACI]]
    return R


def _dibujo_mensula(ax, D, R, cotas=True, barras=False):
    from matplotlib.patches import Polygon, Rectangle, Circle
    h, d, a = D.h, D.d, D.a_carga
    L, hb, lp, rec = D.lm_proy, D.h_borde, D.placa_l, 4.0
    col_w = D.col_b
    db_p = BARRAS[D.barra_ppal][0]
    y0 = -0.45 * h
    ax.add_patch(Rectangle((-col_w, y0), col_w, h - y0 + 30, facecolor="#D5D8DC", edgecolor="#333", lw=1.4))
    ax.add_patch(Polygon([(0, 0), (L, h - hb), (L, h), (0, h)], closed=True, facecolor="#E8ECF1",
                         edgecolor="#333", lw=1.6))
    ax.add_patch(Rectangle((a - lp / 2, h), lp, 2.5, facecolor="#555"))
    ax.annotate("", xy=(a, h + 2.5), xytext=(a, h + 32), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#C0392B"))
    ax.text(a + 2, h + 24, "Vu", color="#C0392B", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(a + 18, h + 1.2), xytext=(a - 2, h + 1.2), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#B0652A"))
    ax.text(a + 6, h + 6, "Nuc", color="#B0652A", fontsize=8.5)
    yt = d                                         # eje del tensor (medido desde el fondo en la cara)
    zz = getattr(R, "stm", {}).get("z", 0.9 * d)
    if barras:
        ldh = R.metr.get("ldh", 30)
        xg = -min(ldh, col_w - rec)
        ax.plot([xg, L - rec - db_p], [yt, yt], color="#C0392B", lw=3)
        ax.plot([xg, xg], [yt, yt - 12 * db_p], color="#C0392B", lw=3)          # gancho de 90° en la columna
        ax.add_patch(Circle((L - rec - db_p / 2, yt), db_p * 0.9, color="#7B241C"))  # barra transversal soldada
        ne, se = R.metr.get("ne", 2), R.metr.get("se", 10)
        for i in range(ne):
            y = yt - (i + 1) * se
            xr = L * (y - 0) / (h - hb) if y < h - hb else L
            xr = min(L, max(4, xr)) - rec
            ax.plot([xg + 3, xr], [y, y], color="#2E8B62", lw=1.6)
        ax.text(L + 4, yt - se - 3, f"Ah: {ne} est. □Ø{D.estribo}\n@ {se:g} cm", color="#2E8B62", fontsize=8)
        ax.text(L + 4, yt + 3, f"Tensor: {R.metr.get('nb', 0)} Ø{D.barra_ppal}\n+ barra transversal soldada",
                color="#C0392B", fontsize=8)
        ax.annotate("", xy=(xg, yt + 6), xytext=(0, yt + 6), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#C0392B"))
        ax.text(xg / 2, yt + 8, f"ldh = {ldh:.0f}", ha="center", fontsize=8, color="#C0392B")
    else:
        ax.plot([-col_w + 5, a], [yt, yt], color="#C0392B", lw=3)
        ax.plot([a, 0], [yt, yt - zz], color="#2F6DB5", lw=3, ls="--")
        ax.text(-col_w + 2, yt + 4, "TENSOR (acero)", color="#C0392B", fontsize=8.5, fontweight="bold")
        ax.text(a * 0.6 + 2, yt - 0.62 * zz, "BIELA\n(concreto)", color="#2F6DB5", fontsize=8.5, fontweight="bold")
    if cotas:
        ax.annotate("", xy=(0, -10), xytext=(a, -10), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(a / 2, -19, f"a = {a:g}", ha="center", fontsize=8.5)
        ax.annotate("", xy=(0, -28), xytext=(L, -28), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(L / 2, -37, f"lm = {L:g}", ha="center", fontsize=8.5)
        xd = L + (55 if barras else 22)
        ax.annotate("", xy=(xd, 0), xytext=(xd, h), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(xd + 2, h * 0.3, f"h = {h:g}", fontsize=8.5)
        ax.annotate("", xy=(xd + 22, h - d), xytext=(xd + 22, h), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#2E8B62"))
        ax.text(xd + 24, h - d / 2, f"d = {d:g}", fontsize=8.5, color="#2E8B62")
        ax.annotate("", xy=(L + 3, h - hb), xytext=(L + 3, h), arrowprops=dict(arrowstyle="<->", lw=0.7, color="#555"))
        ax.text(L + 4, h - hb - 7, f"hb = {hb:g}", fontsize=7.5, color="#555")
        ax.text(a, h + 36, f"placa {lp:g}×{D.placa_b:g}", ha="center", fontsize=7.5, color="#555")
        ax.annotate("", xy=(-col_w, y0 - 8), xytext=(0, y0 - 8), arrowprops=dict(arrowstyle="<->", lw=0.7))
        ax.text(-col_w / 2, y0 - 17, f"columna {col_w:g}", ha="center", fontsize=8)
    ax.set_xlim(-col_w - 12, L + (110 if barras else 60))
    ax.set_ylim(min(y0 - 22, -42), h + 45)
    ax.set_aspect("equal")
    ax.axis("off")


def figuras_mensula(D, R, rec=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, m, b = [], R.metr, R.biela
    # 1. modelo y triángulo de fuerzas
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    _dibujo_mensula(a1, D, R)
    a1.set_title("Modelo de biela y tensor", fontweight="bold")
    T, V = b["T"] - b["Nuc"], D.Vu_sec
    a2.annotate("", xy=(0, -V), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#C0392B"))
    a2.annotate("", xy=(-T, 0), xytext=(0, -V), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#2F6DB5"))
    a2.annotate("", xy=(0, 0), xytext=(-T, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#B03A2E"))
    a2.text(0.6, -V / 2, f"Vu = {V:g} t", color="#C0392B")
    a2.text(-T / 2, 0.08 * V, f"Vu·a/z = {T:.2f} t", color="#B03A2E", ha="center")
    a2.text(-T / 2 - 0.2 * T, -V / 2, f"C = {b['C']:.2f} t", color="#2F6DB5", ha="right")
    a2.set_xlim(-T * 1.5, V * 0.6)
    a2.set_ylim(-V * 1.15, V * 0.25)
    a2.set_aspect("equal")
    a2.axis("off")
    a2.set_title(f"Triángulo de fuerzas (θ = {math.degrees(b['th']):.1f}°)", fontweight="bold")
    fig.tight_layout()
    out.append(("Modelo de biela y tensor y triángulo de fuerzas",
                "La carga baja por la biela de concreto (azul) y se equilibra con el tensor de acero (rojo).", fig))
    # 2. comparación de áreas
    fig, ax = plt.subplots(figsize=(10, 4.2))
    et = ["Af + An", "2/3·Avf + An", "Asc mín", "Biela simplificada", "Puntal-tensor ACI"]
    vals = [m["o1"], m["o2"], m["amin"], m["As_t"], m["As_stm"]]
    cols = ["#2F6DB5", "#2E8B62", "#999999", "#B0652A", "#6E4BB0"]
    bars = ax.bar(et, vals, color=cols)
    ax.axhline(m["Asc"], color="#C0392B", ls="--", lw=2, label=f"Asc de diseño = {m['Asc']:.2f} cm²")
    for r_, v in zip(bars, vals):
        ax.text(r_.get_x() + r_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Área de acero (cm²)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Acero del tensor: cada condición y la que controla", fontweight="bold")
    fig.tight_layout()
    out.append(("Acero principal: condiciones de diseño",
                "Se coloca el mayor valor; los modelos de biela y puntal-tensor dan resultados parecidos.", fig))
    # 3. resistencia de la biela
    fig, ax = plt.subplots(figsize=(10, 3.6))
    et = ["Vn requerido", "0.2·f′c·bw·d", "56·bw·d"]
    vals = [m["Vn"] / 1000, 0.2 * D.fc * D.bw * D.d / 1000, 56 * D.bw * D.d / 1000]
    bars = ax.barh(et[::-1], vals[::-1], color=["#999999", "#2E8B62", "#C0392B"][::-1])
    for r_, v in zip(bars, vals[::-1]):
        ax.text(v, r_.get_y() + r_.get_height() / 2, f"  {v:.1f} t", va="center")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("Fuerza cortante (t)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Verificación de la biela: Vn frente a sus límites", fontweight="bold")
    fig.tight_layout()
    out.append(("Resistencia de la biela de concreto",
                "Vn debe quedar por debajo de los dos límites para que la biela no se aplaste.", fig))
    # 4. demanda / capacidad de todas las verificaciones
    if R.ver:
        out.append(("Resumen de verificaciones (demanda / capacidad)",
                    "Cada barra compara lo que se pide con lo que resiste: verde cumple, rojo no cumple.",
                    _fig_verificaciones(R, "Resumen de verificaciones de la ménsula")))
    out.append(("Detalle de armado de la ménsula", "Tensor con gancho en la columna y barra transversal soldada en "
                "el extremo; estribos cerrados Ah en los 2/3 d superiores.", detalle_mensula(D, R, rec)))
    return out


def detalle_mensula(D, R, rec=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    _dibujo_mensula(ax, D, R, barras=True)
    ax.set_title("DETALLE DE ARMADO DE LA MÉNSULA CORTA", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def svg_mensula(D):
    c = []
    a = max(40, min(260, D.a_carga * 6))
    c.append('<rect x="90" y="20" width="90" height="240" fill="#BFC5CE" stroke="#1E2A3A" stroke-width="2"/>')
    c.append(f'<polygon points="180,90 {180 + a + 90},90 {180 + a + 90},140 180,190" fill="#D6DCE4" stroke="#1E2A3A" stroke-width="2"/>')
    c.append(_flecha(180 + a, 20, 180 + a, 86, "#C0392B", 3))
    c.append(f'<text x="{190 + a}" y="40" class="v">Vu = {D.Vu_sec:g} t</text>')
    c.append(_flecha(180 + a - 10, 84, 180 + a + 50, 84, "#B0652A", 2))
    c.append(f'<text x="{190 + a + 40}" y="76" class="t">Nuc = {max(D.Nuc, 0.2 * D.Vu_sec):.2f} t</text>')
    c.append(f'<line x1="100" y1="100" x2="{180 + a}" y2="100" stroke="#C0392B" stroke-width="4"/>')
    c.append(f'<line x1="{180 + a}" y1="100" x2="182" y2="180" stroke="#2F6DB5" stroke-width="4" stroke-dasharray="8 5"/>')
    c.append('<text x="196" y="118" class="s">tensor</text><text x="196" y="165" class="s">biela</text>')
    c.append(_cota(180, 180 + a, 215, f"a = {D.a_carga:g} cm"))
    c.append(f'<text x="{200 + a + 90}" y="120" class="t">h = {D.h:g}</text>')
    c.append('<text x="320" y="262" text-anchor="middle" class="s">Ménsula corta: el corte lo toma una biela, no los estribos</text>')
    return _svg("".join(c), 640, 270 if D.modo != 3 else 268)


# ===========================================================================
# PLANOS (lámina A3) Y VISTA 3D DE LA ARMADURA
# ===========================================================================
AZ_EST, AZ_MONT, C_LONG, C_CONC = "#C0392B", "#2E86C1", "#1C1C1C", "#BFC5CC"


def cuadro_estribos(D, R):
    """Filas del cuadro de estribos (o de refuerzo de la ménsula), con encabezado."""
    m = R.metr
    if D.modo == 5:
        return [["Refuerzo", "Cantidad", "Diámetro", "Ubicación"],
                ["Tensor", f"{m['nb']}", f"Ø{D.barra_ppal}", "Cara superior"],
                ["Barra transversal", "1", f"Ø{D.barra_ppal}", "Soldada en el extremo"],
                ["Estribos Ah", f"{m['ne']}", f"Ø{D.estribo}", f"2/3 d sup. @ {m['se']:g} cm"]]
    cuadro = [["Tramo", "Cantidad", "Espaciamiento", "Desde (cm)", "Hasta (cm)"]]
    if D.modo == 4 and R.conf_col:
        cc = R.conf_col
        x_end = 5 + cc["n0"] * cc["s0"]
        cuadro += [["Primer estribo", "1", "—", "0", "5"],
                   ["Zona de confinamiento Lo", f"{cc['n0']}", f"{cc['s0']:g} cm", "5", f"{x_end:g}"],
                   ["Zona central (resto)", "—", f"{cc['sc']:g} cm", f"{x_end:g}", f"{D.ln_col * 100 - x_end:g}"]]
    elif D.modo == 4:
        cuadro.append(["Todo el elemento", "—", f"{R.tramos[0][1]:g} cm", "—", "—"])
    else:
        cuadro.append(["Primer estribo", "1", "—", "0", f"{R.primero:g}"])
        x, i = R.primero, 1
        half = longitud_viga(D) / 2
        for n, s in R.tramos:
            if n == "resto":
                cnt = int((half - x) // s)
                cuadro.append([f"Tramo {i} (resto)", f"{cnt}", f"{s:g} cm", f"{x:g}", f"{x + cnt * s:g}"])
                x += cnt * s
                break
            cuadro.append([f"Tramo {i}", f"{n}", f"{s:g} cm", f"{x:g}", f"{x + n * s:g}"])
            x += n * s
            i += 1
        pos, _ = posiciones_estribos(D, R)
        nm = sum(1 for _, t in pos if t == "montaje")
        if nm:
            cuadro.append(["Montaje (centro)", f"{nm}", f"{R.smax:g} cm", f"{x:g}", f"{x + nm * R.smax:g}"])
    return cuadro


def armado(D, R, rec=4.0):
    """Geometría de la armadura (cm) para los planos y la vista 3D.
    x: eje del elemento; u: ancho de la sección (0..bw); v: altura (0..h)."""
    dbe = BARRAS[D.estribo][0]
    db = max(D.db_long, 1.27)
    Ab = math.pi * db ** 2 / 4
    G = dict(tipo="viga", bw=D.bw, h=D.h, rec=rec, dbe=dbe, db=db, est=[], ref=True)
    if D.modo == 5:
        G["tipo"] = "mensula"
        return G
    if D.modo == 4:
        G["tipo"] = "columna" if R.conf_col else "tramo"
        if R.conf_col:
            cc = R.conf_col
            L = D.ln_col * 100
            xs = [5 + i * cc["s0"] for i in range(cc["n0"] + 1)]
            xe = xs[-1]
            xm = list(np.arange(xe + cc["sc"], L - xe - cc["sc"] / 2 + 1e-6, cc["sc"]))
            est = [(x, "calculo") for x in xs] + [(x, "calculo") for x in xm] + [(L - x, "calculo") for x in xs[::-1]]
        else:
            L = 300.0
            s = R.tramos[0][1]
            est = [(float(x), "calculo") for x in np.arange(s / 2, L, s)]
        nb_cara = 3 if max(D.bw, D.h) > 45 else 2
        G.update(L=L, est=est, n_sup=nb_cara, n_inf=nb_cara, n_lat=nb_cara - 2,
                 vertical=bool(R.conf_col) or D.Nu >= 0)      # compresión: columna (vertical); tracción: viga
        return G
    L = longitud_viga(D)
    pos, s_ult = posiciones_estribos(D, R)
    est = list(pos)
    half = L / 2
    en_centro = abs(pos[-1][0] - half) < 1e-6
    if (not en_centro) and 2 * (half - pos[-1][0]) > s_ult + 1e-6:
        est.append((half, "montaje"))
    est += [(L - x, t) for x, t in reversed(pos) if abs(L - x - x) > 1e-6]
    if D.modo == 2:
        ns = max(2, round(D.As_sup / Ab))
        ni = max(2, round(D.As_inf / Ab))
        G["ref"] = False
    else:
        ns, ni = 2, 2
    G.update(L=L, est=est, n_sup=ns, n_inf=ni, n_lat=0)
    return G


def _pos_barras(G):
    """Centros (u, v) de las barras longitudinales en la sección."""
    c = G["rec"] + G["dbe"] + G["db"] / 2
    b, h = G["bw"], G["h"]
    out = []
    for n, v in ((G["n_inf"], c), (G["n_sup"], h - c)):
        for i in range(n):
            out.append((c + (b - 2 * c) * i / max(n - 1, 1), v))
    for i in range(G.get("n_lat", 0)):          # barras intermedias en las caras laterales (columna)
        v = c + (h - 2 * c) * (i + 1) / (G["n_lat"] + 1)
        out += [(c, v), (b - c, v)]
    return out


def _poli_estribo(G, gk):
    """Polilínea (u, v) del estribo cerrado con dos ganchos a 135° en la esquina superior izquierda."""
    c = G["rec"] + G["dbe"] / 2
    b, h = G["bw"], G["h"]
    k = gk / math.sqrt(2)
    return [(c + k, h - c - k * 1.08), (c, h - c), (b - c, h - c), (b - c, c), (c, c), (c, h - c),
            (c + k * 1.08, h - c - k)]


# ---------------------------------------------------------------- utilidades de dibujo
def _cota_pl(ax, p1, p2, txt, off=0.0, vertical=False, fs=7.5, color="#222"):
    """Línea de cota con flechas y texto (off desplaza la línea perpendicularmente)."""
    (x1, y1), (x2, y2) = p1, p2
    if vertical:
        x1 = x2 = x1 + off
    else:
        y1 = y2 = y1 + off
    ax.annotate("", xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color=color,
                                                                  mutation_scale=7, shrinkA=0, shrinkB=0))
    if vertical:
        ax.text(x1 - 1.5, (y1 + y2) / 2, txt, rotation=90, ha="right", va="center", fontsize=fs, color=color)
    else:
        ax.text((x1 + x2) / 2, y1 + 1.2, txt, ha="center", va="bottom", fontsize=fs, color=color)


def _tabla_ax(ax, filas, anchos, titulo=None, fs=7.5):
    """Dibuja una tabla con bordes en un eje sin coordenadas (0..1)."""
    from matplotlib.patches import Rectangle
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = 0.88 if titulo else 1.0
    if titulo:
        ax.text(0.0, 0.97, titulo, fontsize=fs + 1.5, fontweight="bold", va="top")
    n = len(filas)
    hrow = min(0.12, top / max(n, 1))
    tot = sum(anchos)
    xs = [0]
    for a in anchos:
        xs.append(xs[-1] + a / tot)
    for i, fila in enumerate(filas):
        y = top - (i + 1) * hrow
        if i == 0:
            ax.add_patch(Rectangle((0, y), 1, hrow, facecolor="#DCE5F2", edgecolor="none"))
        for j, celda in enumerate(fila):
            ax.text((xs[j] + xs[j + 1]) / 2, y + hrow / 2, str(celda), ha="center", va="center", fontsize=fs,
                    fontweight="bold" if i == 0 else "normal")
        ax.plot([0, 1], [y, y], color="#555", lw=0.5)
    ax.plot([0, 1], [top, top], color="#555", lw=0.5)
    for x in xs:
        ax.plot([x, x], [top - n * hrow, top], color="#555", lw=0.5)


def _cajetin(fig, D, info, rect, lamina="E-01", escala="Indicada"):
    import datetime
    from matplotlib.patches import Rectangle
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 100, 40, fill=False, lw=1.2))
    for y in (30, 20, 10):
        ax.plot([0, 100], [y, y], lw=0.6, color="#333")
    ax.plot([60, 60], [0, 20], lw=0.6, color="#333")
    ax.plot([80, 80], [0, 10], lw=0.6, color="#333")
    ax.text(50, 35, "UNIVERSIDAD NACIONAL DE SAN CRISTÓBAL DE HUAMANGA", ha="center", va="center", fontsize=7.5,
            fontweight="bold")
    ax.text(50, 32, "Escuela Profesional de Ingeniería Civil", ha="center", va="center", fontsize=6.5)
    ax.text(2, 27.5, "PROYECTO:", fontsize=5.5, va="top", color="#555")
    ax.text(50, 24, info.get("proyecto", ""), ha="center", va="center", fontsize=7.5)
    ax.text(2, 17.5, "PLANO:", fontsize=5.5, va="top", color="#555")
    el = info.get("elemento", "") or D.titulo
    ax.text(30, 13.5, f"Armado por corte: {el}"[:48], ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(62, 17.5, "LÁMINA:", fontsize=5.5, va="top", color="#555")
    ax.text(80, 13.5, lamina, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(2, 7.5, "ELABORADO:", fontsize=5.5, va="top", color="#555")
    ax.text(30, 3.8, info.get("elaborado", "") or "—", ha="center", va="center", fontsize=6.5)
    ax.text(62, 7.5, "ESCALA:", fontsize=5.5, va="top", color="#555")
    ax.text(70, 3.8, escala, ha="center", va="center", fontsize=6.5)
    ax.text(82, 7.5, "FECHA:", fontsize=5.5, va="top", color="#555")
    ax.text(90, 3.8, datetime.date.today().strftime("%d/%m/%Y"), ha="center", va="center", fontsize=6.5)
    rv = info.get("revisado", "")
    if rv:
        ax.text(30, 8.6, f"Revisó: {rv}"[:44], ha="center", va="center", fontsize=5.5, color="#555")


def _seccion_plano(ax, D, G, titulo, nota):
    """Corte transversal acotado con barras longitudinales, estribo y gancho."""
    from matplotlib.patches import Rectangle, Circle
    b, h, r = G["bw"], G["h"], G["rec"]
    ax.add_patch(Rectangle((0, 0), b, h, facecolor="#F4F5F6", edgecolor="black", lw=1.4, hatch="....",
                           alpha=1))
    gk = gancho(D)[0]
    pts = _poli_estribo(G, gk)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=AZ_EST, lw=1.8, solid_joinstyle="round")
    for (u, v) in _pos_barras(G):
        ax.add_patch(Circle((u, v), G["db"] / 2, color=C_LONG, zorder=3))
    _cota_pl(ax, (0, 0), (b, 0), f"{b:g}", off=-0.18 * h)
    _cota_pl(ax, (0, 0), (0, h), f"{h:g}", off=-0.16 * b - 2, vertical=True)
    ax.text(b + 2, r, f"r = {r:g} cm", fontsize=6.5, va="bottom")
    ax.text(b + 2, h * 0.55, f"□ Ø{D.estribo}\n{nota}", fontsize=7, va="center", color=AZ_EST)
    tl = f"{G['n_sup']} Ø{_frac_db(G['db'])}" + (" (ref.)" if G["ref"] else "")
    ti = f"{G['n_inf']} Ø{_frac_db(G['db'])}" + (" (ref.)" if G["ref"] else "")
    ax.text(b / 2, h + 0.06 * h + 1, tl, ha="center", fontsize=6.5)
    ax.text(b / 2, -0.36 * h, ti, ha="center", fontsize=6.5, va="top")
    ax.text(b + 2, h - r - gk * 0.5, f"gancho 135°\next. {gk:.1f}", fontsize=6, va="top", color="#555")
    ax.set_title(titulo, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlim(-0.45 * b - 6, b + 0.9 * b + 8)
    ax.set_ylim(-0.5 * h, h * 1.22)
    ax.set_aspect("equal")
    ax.axis("off")


def _frac_db(db):
    for k, (d_, _) in BARRAS.items():
        if abs(d_ - db) < 0.05:
            return k
    return f"{db:.2f} cm"


def _detalle_estribo(ax, D, G):
    """Estribo desarrollado con medidas interiores, ganchos y longitud de corte."""
    r, b, h = G["rec"], G["bw"], G["h"]
    gk = gancho(D)[0]
    bi, hi = b - 2 * r, h - 2 * r
    Lc = 2 * bi + 2 * hi + 2 * gk
    k = gk / math.sqrt(2)
    ax.plot([0, bi, bi, 0, 0], [0, 0, hi, hi, 0], color=AZ_EST, lw=2.2, solid_joinstyle="round")
    ax.plot([0, k], [hi, hi - k * 1.08], color=AZ_EST, lw=2.2)
    ax.plot([0, k * 1.08], [hi, hi - k], color=AZ_EST, lw=2.2)
    _cota_pl(ax, (0, 0), (bi, 0), f"{bi:g}", off=-0.2 * hi - 2)
    _cota_pl(ax, (0, 0), (0, hi), f"{hi:g}", off=-0.2 * bi - 3, vertical=True)
    ax.text(k + 1.5, hi - k - 1, f"{gk:.1f}", fontsize=7, color="#333")
    ax.text(bi / 2, hi / 2, f"□ Ø{D.estribo}\nLc = {Lc:.1f} cm\n135° ({gancho(D)[1]}db)",
            ha="center", va="center", fontsize=7.5)
    ax.set_title("DETALLE DEL ESTRIBO (medidas interiores, cm)", fontsize=9, fontweight="bold", pad=4)
    ax.set_xlim(-0.55 * bi - 6, bi * 1.25 + 4)
    ax.set_ylim(-0.5 * hi - 4, hi * 1.2 + 4)
    ax.set_aspect("equal")
    ax.axis("off")


def _elevacion_plano(ax, D, R, G):
    """Elevación acotada (viga) a escala, con ejes de corte A-A y B-B."""
    from matplotlib.patches import Rectangle
    L, H, r = G["L"], G["h"], G["rec"]
    k = max(1, min(3, round((L / H) / 10)))                  # exageración vertical para que se lea
    Hd = H * k
    sc = lambda v: v * k
    ax.add_patch(Rectangle((0, 0), L, Hd, facecolor="#F4F5F6", edgecolor="black", lw=1.4))
    if D.modo == 1:                       # apoyos simples (articulado y deslizante)
        from matplotlib.patches import Polygon
        t = 0.35 * Hd
        for xc in (0, L):
            ax.add_patch(Polygon([(xc, 0), (xc - 0.6 * t, -t), (xc + 0.6 * t, -t)], closed=True, facecolor="white",
                                 edgecolor="#333", lw=1.2))
        ax.plot([L - 0.9 * t, L + 0.9 * t], [-t - 0.15 * t] * 2, color="#333", lw=1)
    else:
        for xc in (-30, L):
            ax.add_patch(Rectangle((xc, -0.7 * Hd), 30, 2.4 * Hd, facecolor="#D5D8DC", edgecolor="#555", lw=1,
                                   hatch="////"))
    zc = getattr(R, "zona_conf", 0) or 0
    if zc:
        for a_, b_ in ((0, zc), (L - zc, L)):
            ax.add_patch(Rectangle((a_, 0), b_ - a_, Hd, facecolor="#6E4BB0", alpha=0.08, lw=0))
        ax.text(zc / 2, Hd + 0.05 * Hd, f"2h = {zc:g}", ha="center", fontsize=6.5, color="#6E4BB0")
    c = r + G["dbe"] + G["db"] / 2
    ext_ = 4 if D.modo == 1 else 25
    for v in (c, H - c):
        ax.plot([ext_ - 8 if D.modo == 1 else -25, L - ext_ + 8 if D.modo == 1 else L + 25], [sc(v), sc(v)],
                color=C_LONG, lw=1.4)
    for x, t in G["est"]:
        ax.plot([x, x], [sc(r), sc(H - r)], color=AZ_EST if t == "calculo" else AZ_MONT, lw=0.9,
                ls="-" if t == "calculo" else (0, (3, 2)))
    ax.axvline(L / 2, color="#888", ls="-.", lw=0.7)
    ax.text(L / 2, 2.0 * Hd, "CL", ha="center", fontsize=7, color="#555")
    # cadena de cotas (mitad izquierda y su simétrica)
    yb = -0.45 * Hd
    x = 0.0
    tramos = [(R.primero, f"1@{R.primero:g}")]
    for n, s in R.tramos:
        if n == "resto":
            tramos.append((None, f"resto @ {s:g}"))
            break
        tramos.append((n * s, f"{n}@{s:g}"))
    x = 0.0
    for long_, txt in tramos:
        if long_ is None:
            long_ = L / 2 - x
        if long_ <= 0:
            continue
        for a_, b_ in ((x, x + long_), (L - x - long_, L - x)):
            ax.annotate("", xy=(a_, yb), xytext=(b_, yb),
                        arrowprops=dict(arrowstyle="|-|", lw=0.6, mutation_scale=3, shrinkA=0, shrinkB=0))
            ax.text((a_ + b_) / 2, yb - 0.06 * Hd, txt, ha="center", va="top", fontsize=6.5,
                    rotation=90 if long_ < 0.045 * L else 0)
        x += long_
    pos = posiciones_estribos(D, R)[0]
    if pos and pos[-1][1] == "montaje" and x < L / 2:
        ax.text(L / 2, yb - 0.06 * Hd, f"montaje @ {R.smax:g}", ha="center", va="top", fontsize=6.5, color=AZ_MONT)
    _cota_pl(ax, (0, 1.55 * Hd), (L, 1.55 * Hd), f"{'L' if D.modo == 1 else 'ln'} = {L / 100:.2f} m", fs=8)
    # cortes A-A (zona de apoyo) y B-B (centro)
    xa = min(R.primero + 0.5 * (R.tramos[0][1] if R.tramos and R.tramos[0][0] != "resto" else 20), L * 0.15)
    for xx, lab in ((xa, "A"), (L / 2 - 0.08 * L, "B")):
        ax.plot([xx, xx], [-0.25 * Hd, 1.25 * Hd], color="#111", lw=0.9, ls=(0, (8, 3, 2, 3)))
        for yy in (-0.25 * Hd, 1.25 * Hd):
            ax.text(xx + 0.008 * L, yy, lab, fontsize=8, fontweight="bold", va="center")
    if D.modo != 1:
        ax.text(-30, 1.15 * Hd, "Columna", fontsize=6.5, rotation=90, va="bottom")
    txt_esc = f"(escala vertical ×{k})" if k > 1 else ""
    ax.set_title(f"ELEVACIÓN DE LA VIGA — distribución de estribos (cm) {txt_esc}", fontsize=10,
                 fontweight="bold", loc="left")
    ax.set_xlim(-45, L + 45)
    ax.set_ylim(-1.0 * Hd, 1.9 * Hd)
    ax.set_aspect("equal")
    ax.axis("off")


def _elevacion_columna(ax, D, R, G):
    """Elevación vertical de la columna con zonas Lo."""
    from matplotlib.patches import Rectangle
    L, b, r = G["L"], G["bw"], G["rec"]
    ax.add_patch(Rectangle((0, 0), b, L, facecolor="#F4F5F6", edgecolor="black", lw=1.4))
    for y0 in (-40, L):
        ax.add_patch(Rectangle((-0.8 * b, y0), 2.6 * b, 40, facecolor="#D5D8DC", edgecolor="#555", hatch="////"))
    c = r + G["dbe"] + G["db"] / 2
    for u in (c, b - c):
        ax.plot([u, u], [-35, L + 35], color=C_LONG, lw=1.4)
    for x, _ in G["est"]:
        ax.plot([r, b - r], [x, x], color=AZ_EST, lw=0.9)
    if R.conf_col:
        cc = R.conf_col
        for a_, b_ in ((0, cc["Lo"]), (L - cc["Lo"], L)):
            ax.add_patch(Rectangle((0, a_), b, b_ - a_, facecolor="#6E4BB0", alpha=0.08, lw=0))
            _cota_pl(ax, (b, a_), (b, b_), f"Lo={cc['Lo']:.0f}", off=0.35 * b + 6, vertical=True, fs=6.5,
                  color="#6E4BB0")
        xe = 5 + cc["n0"] * cc["s0"]
        ax.text(-0.25 * b, xe / 2, f"{cc['n0']}@{cc['s0']:g}\n1@5", ha="right", va="center", fontsize=6.5)
        ax.text(-0.25 * b, L / 2, f"resto\n@{cc['sc']:g}", ha="right", va="center", fontsize=6.5)
        ax.text(-0.25 * b, L - xe / 2, f"1@5\n{cc['n0']}@{cc['s0']:g}", ha="right", va="center", fontsize=6.5)
    _cota_pl(ax, (b, 0), (b, L), f"ln = {L / 100:.2f} m", off=0.9 * b + 12, vertical=True, fs=7.5)
    ax.plot([b / 2 - 0.9 * b, b / 2 + 0.9 * b], [L / 2 + 0.1 * L] * 2, color="#111", lw=0.9, ls=(0, (8, 3, 2, 3)))
    ax.text(b / 2 + 0.95 * b, L / 2 + 0.1 * L, "A", fontsize=8, fontweight="bold", va="center")
    ax.set_title("ELEVACIÓN DE LA COLUMNA", fontsize=10, fontweight="bold")
    ax.set_xlim(-1.6 * b - 10, 2.6 * b + 20)
    ax.set_ylim(-50, L + 50)
    ax.set_aspect("equal")
    ax.axis("off")


def _especificaciones(ax, D, R, G, rec):
    ax.axis("off")
    gk, kk, sis = gancho(D)
    lin = ["ESPECIFICACIONES TÉCNICAS",
           f"Concreto: f′c = {D.fc:g} kg/cm²" + (" (liviano)" if D.lam < 1 else ""),
           f"Acero: fy = {D.fy:g} kg/cm² (ASTM A615 grado 60)",
           f"Recubrimiento libre: r = {rec:g} cm",
           f"Estribos cerrados con ganchos a 135°, extensión {kk}db = {gk:.1f} cm"
           + (" (gancho sísmico, E.060 21.1)" if sis else " (E.060 7.1.3)"),
           "Alternar la posición de los ganchos en estribos consecutivos.",
           "Medidas en cm, salvo indicación.",
           "Norma: E.060 Concreto Armado (2009)."]
    if G.get("ref"):
        lin.append("Acero longitudinal referencial: se diseña por flexión.")
    for i, t in enumerate(lin):
        ax.text(0, 1 - i * 0.105, t, fontsize=8.5 if i == 0 else 7.2, fontweight="bold" if i == 0 else "normal",
                va="top", transform=ax.transAxes)


def plano_lamina(D, R, rec=4.0, info=None):
    """Lámina A3 apaisada con elevación, cortes, detalle del estribo, cuadros, especificaciones y cajetín."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    info = info or {}
    fig = plt.figure(figsize=(16.54, 11.69))
    fig.patches.append(Rectangle((0.012, 0.015), 0.976, 0.97, transform=fig.transFigure,
                                                    fill=False, lw=1.6, color="black"))
    G = armado(D, R, rec)
    M = metrado(D, R, rec)
    if G["tipo"] == "mensula":
        _plano_mensula(fig, D, R, G, rec, M, info)
    elif G["tipo"] in ("columna", "tramo"):
        ax = fig.add_axes([0.03, 0.06, 0.27, 0.88])
        if R.conf_col:
            _elevacion_columna(ax, D, R, G)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Sección con carga axial\n(sin altura libre ln: se detalla la sección\ny el espaciamiento uniforme)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        nota = (f"@ {R.conf_col['s0']:g} en Lo\n@ {R.conf_col['sc']:g} resto" if R.conf_col
                else f"@ {R.tramos[0][1]:g}")
        _seccion_plano(fig.add_axes([0.31, 0.50, 0.30, 0.42]), D, G, "CORTE A-A", nota)
        _detalle_estribo(fig.add_axes([0.31, 0.08, 0.30, 0.38]), D, G)
        _tabla_ax(fig.add_axes([0.63, 0.66, 0.35, 0.28]), cuadro_estribos(D, R), [3.4, 1.4, 1.8, 1.4, 1.4],
                  "CUADRO DE ESTRIBOS (desde cada extremo)")
        _tabla_ax(fig.add_axes([0.63, 0.47, 0.35, 0.17]), _filas_metrado(D, M), [1.2, 1.2, 1.4, 1.4, 1.4],
                  "METRADO DEL ACERO TRANSVERSAL")
        _especificaciones(fig.add_axes([0.63, 0.22, 0.35, 0.22]), D, R, G, rec)
        _cajetin(fig, D, info, [0.63, 0.025, 0.35, 0.17])
    else:
        _elevacion_plano(fig.add_axes([0.03, 0.50, 0.94, 0.46]), D, R, G)
        s_apoyo = R.tramos[0][1] if R.tramos and R.tramos[0][0] != "resto" else R.tramos[0][1]
        s_centro = R.smax if (posiciones_estribos(D, R)[0][-1][1] == "montaje") else R.tramos[-1][1]
        _seccion_plano(fig.add_axes([0.02, 0.06, 0.20, 0.40]), D, G, "CORTE A-A", f"@ {s_apoyo:g}")
        _seccion_plano(fig.add_axes([0.21, 0.06, 0.20, 0.40]), D, G, "CORTE B-B", f"@ {s_centro:g}")
        _detalle_estribo(fig.add_axes([0.41, 0.24, 0.21, 0.24]), D, G)
        _tabla_ax(fig.add_axes([0.41, 0.03, 0.21, 0.19]), _filas_metrado(D, M), [1.2, 1.2, 1.4, 1.4, 1.4],
                  "METRADO (por viga)", fs=6.8)
        _tabla_ax(fig.add_axes([0.63, 0.30, 0.35, 0.18]), cuadro_estribos(D, R), [3.0, 1.4, 1.9, 1.4, 1.4],
                  "CUADRO DE ESTRIBOS (cada extremo, desde la cara)", fs=7)
        _especificaciones(fig.add_axes([0.63, 0.20, 0.35, 0.10]), D, R, G, rec)
        _cajetin(fig, D, info, [0.63, 0.025, 0.35, 0.155])
    return fig


def _filas_metrado(D, M):
    return [["Ø", "Cant.", "Lc (m)", "Long. (m)", "Peso (kg)"],
            [D.estribo, f"{M['n']}", f"{M['Lc'] / 100:.2f}", f"{M['L']:.2f}", f"{M['kg']:.2f}"],
            ["", "", "", "Total", f"{M['kg']:.2f}"]]


def _plano_mensula(fig, D, R, G, rec, M, info):
    from matplotlib.patches import Rectangle, Circle
    m = R.metr
    ax = fig.add_axes([0.03, 0.40, 0.55, 0.56])
    _dibujo_mensula(ax, D, R, cotas=True, barras=True)
    ax.set_title("ELEVACIÓN DE LA MÉNSULA", fontsize=10, fontweight="bold", loc="left")
    # planta
    ax = fig.add_axes([0.03, 0.04, 0.30, 0.34])
    lm, bw = D.lm_proy, D.bw
    db_p = BARRAS[D.barra_ppal][0]
    ldh = m["ldh"]
    ax.add_patch(Rectangle((-D.col_b, -10), D.col_b, bw + 20, facecolor="#D5D8DC", edgecolor="#555", hatch="////"))
    ax.add_patch(Rectangle((0, 0), lm, bw, facecolor="#F4F5F6", edgecolor="black", lw=1.4))
    ax.add_patch(Rectangle((D.a_carga - D.placa_l / 2, (bw - D.placa_b) / 2), D.placa_l, D.placa_b,
                           facecolor="#888", alpha=0.35, edgecolor="#333"))
    nb = m["nb"]
    c = rec + G["dbe"] + db_p / 2
    for i in range(nb):
        y = c + (bw - 2 * c) * i / max(nb - 1, 1)
        ax.plot([-min(ldh, D.col_b - rec), lm - rec - db_p], [y, y], color=AZ_EST, lw=2)
    ax.plot([lm - rec - db_p / 2] * 2, [rec, bw - rec], color="#7B241C", lw=3)
    _cota_pl(ax, (0, 0), (lm, 0), f"lm = {lm:g}", off=-12)
    _cota_pl(ax, (lm, 0), (lm, bw), f"bw = {bw:g}", off=10, vertical=True)
    ax.text(D.a_carga, bw + 4, "placa", ha="center", fontsize=6.5)
    ax.text(lm + 2, bw + 6, "barra transversal\nsoldada", fontsize=6.5, color="#7B241C")
    ax.set_title("PLANTA (tensor y barra transversal)", fontsize=9, fontweight="bold")
    ax.set_xlim(-D.col_b - 8, lm + 40)
    ax.set_ylim(-26, bw + 22)
    ax.set_aspect("equal")
    ax.axis("off")
    # corte en la cara de la columna
    ax = fig.add_axes([0.33, 0.04, 0.26, 0.34])
    h = D.h
    ax.add_patch(Rectangle((0, 0), bw, h, facecolor="#F4F5F6", edgecolor="black", lw=1.4, hatch="...."))
    for i in range(nb):
        u = c + (bw - 2 * c) * i / max(nb - 1, 1)
        ax.add_patch(Circle((u, D.d), db_p / 2, color=AZ_EST, zorder=3))
    for i in range(m["ne"]):
        v = D.d - (i + 1) * m["se"]
        ax.plot([rec, bw - rec], [v, v], color="#2E8B62", lw=1.6)
        for u in (rec, bw - rec):
            ax.add_patch(Circle((u, v), G["dbe"] / 2 + 0.3, color="#2E8B62"))
    _cota_pl(ax, (0, 0), (bw, 0), f"{bw:g}", off=-8)
    _cota_pl(ax, (0, 0), (0, h), f"h = {h:g}", off=-8, vertical=True)
    ax.text(bw + 3, D.d, f"{nb} Ø{D.barra_ppal}", fontsize=7, color=AZ_EST, va="center")
    ax.text(bw + 3, D.d - m["se"], f"Ah: {m['ne']} □Ø{D.estribo}\n@ {m['se']:g}", fontsize=7, color="#2E8B62",
            va="top")
    ax.set_title("CORTE A-A (cara de la columna)", fontsize=9, fontweight="bold")
    ax.set_xlim(-22, bw + 30)
    ax.set_ylim(-18, h + 8)
    ax.set_aspect("equal")
    ax.axis("off")
    _tabla_ax(fig.add_axes([0.62, 0.70, 0.36, 0.25]), cuadro_estribos(D, R), [2.2, 1.0, 1.1, 2.4],
              "CUADRO DE REFUERZO DE LA MÉNSULA")
    Lt = (min(ldh, D.col_b - rec) + lm - rec - db_p + 12 * db_p) / 100
    pt = PESO_BARRA.get(D.barra_ppal, 1.552)
    filas = [["Elemento", "Ø", "Cant.", "Long. (m)", "Peso (kg)"],
             ["Tensor", D.barra_ppal, f"{nb}", f"{nb * Lt:.2f}", f"{nb * Lt * pt:.2f}"],
             ["Barra transv.", D.barra_ppal, "1", f"{(bw - 2 * rec) / 100:.2f}", f"{(bw - 2 * rec) / 100 * pt:.2f}"],
             ["Estribos Ah", D.estribo, f"{M['n']}", f"{M['L']:.2f}", f"{M['kg']:.2f}"]]
    _tabla_ax(fig.add_axes([0.62, 0.50, 0.36, 0.18]), filas, [1.8, 1.0, 0.9, 1.2, 1.2], "METRADO (por ménsula)")
    ax = fig.add_axes([0.62, 0.22, 0.36, 0.26])
    _especificaciones(ax, D, R, G, rec)
    ax.text(0, 0.02, f"Tensor anclado en la columna con gancho a 90° (ldh = {ldh:.0f} cm).\n"
                     f"Placa de apoyo {D.placa_l:g} × {D.placa_b:g} cm.", fontsize=7.2, transform=ax.transAxes)
    _cajetin(fig, D, info, [0.62, 0.025, 0.36, 0.17])


# ---------------------------------------------------------------- vista 3D
def _segmentos_3d(D, R, rec=4.0):
    """Líneas 3D (x, y, z) agrupadas por tipo: concreto (cajas), barras longitudinales, estribos."""
    G = armado(D, R, rec)
    gk = gancho(D)[0]
    cajas, long_, est_c, est_m, otros = [], [], [], [], []
    if G["tipo"] == "mensula":
        m = R.metr
        h, d, bw, lm, hb = D.h, D.d, D.bw, D.lm_proy, D.h_borde
        cb = D.col_b
        y0 = -0.6 * h
        cajas.append(("box", (-cb, 0), (0, bw), (y0, h + 40), C_CONC))
        cajas.append(("prisma", [(0, 0), (lm, h - hb), (lm, h), (0, h)], (0, bw), "#AEB6BF"))
        cajas.append(("box", (D.a_carga - D.placa_l / 2, D.a_carga + D.placa_l / 2),
                      ((bw - D.placa_b) / 2, (bw + D.placa_b) / 2), (h, h + 2), "#555"))
        db_p = BARRAS[D.barra_ppal][0]
        c = rec + G["dbe"] + db_p / 2
        xg = -min(m["ldh"], cb - rec)
        for i in range(m["nb"]):
            y = c + (bw - 2 * c) * i / max(m["nb"] - 1, 1)
            long_.append([(xg, y, d - 12 * db_p), (xg, y, d), (lm - rec - db_p, y, d)])
        otros.append([(lm - rec - db_p / 2, rec, d), (lm - rec - db_p / 2, bw - rec, d)])
        for i in range(m["ne"]):
            z = d - (i + 1) * m["se"]
            xr = lm * z / (h - hb) if z < h - hb else lm
            xr = min(lm, max(4, xr)) - rec
            x0 = xg + 3
            est_c.append([(x0, rec, z), (xr, rec, z), (xr, bw - rec, z), (x0, bw - rec, z), (x0, rec, z)])
        for u in (rec + 2, bw - rec - 2):          # barras de la columna (referenciales)
            for x in (-cb + rec + 2, -rec - 2):
                long_.append([(x, u, y0), (x, u, h + 40)])
        return dict(cajas=cajas, long=long_, est_c=est_c, est_m=est_m, otros=otros, G=G)
    poli = _poli_estribo(G, gk)
    barras = _pos_barras(G)
    L = G["L"]
    col = G.get("vertical", False)
    # mapeo (eje, u, v) -> (x, y, z); la columna se dibuja vertical
    P = (lambda s, u, v: (u, v, s)) if col else (lambda s, u, v: (s, u, v))
    if col:
        cajas.append(("box", (0, G["bw"]), (0, G["h"]), (0, L), C_CONC))
        if G["tipo"] == "columna":
            for z0 in (-40, L):
                cajas.append(("box", (-0.6 * G["bw"], 1.6 * G["bw"]), (-0.6 * G["h"], 1.6 * G["h"]), (z0, z0 + 40), "#AEB6BF"))
        ext = 35 if G["tipo"] == "columna" else 0
    elif D.modo == 1:                      # viga simplemente apoyada: apoyos puntuales, no columnas
        cajas.append(("box", (-15, L + 15), (0, G["bw"]), (0, G["h"]), C_CONC))
        for x0 in (0, L):
            cajas.append(("box", (x0 - 15, x0 + 15), (-8, G["bw"] + 8), (-0.9 * G["h"], -1), "#AEB6BF"))
        ext = 11
    else:
        cajas.append(("box", (0, L), (0, G["bw"]), (0, G["h"]), C_CONC))
        if G["tipo"] == "viga":
            for x0 in (-30, L):
                cajas.append(("box", (x0, x0 + 30), (-10, G["bw"] + 10), (-0.6 * G["h"], 1.6 * G["h"]), "#AEB6BF"))
        ext = 25 if G["tipo"] == "viga" else 0
    for (u, v) in barras:
        long_.append([P(-ext, u, v), P(L + ext, u, v)])
    for x, t in G["est"]:
        (est_c if t == "calculo" else est_m).append([P(x, u, v) for (u, v) in poli])
    return dict(cajas=cajas, long=long_, est_c=est_c, est_m=est_m, otros=otros, G=G)


def figura_3d(D, R, rec=4.0):
    """Figura 3D interactiva (plotly). Devuelve None si plotly no está instalado."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    S = _segmentos_3d(D, R, rec)
    fig = go.Figure()

    def caja_xyz(xr, yr, zr):
        X, Y, Z = [], [], []
        for x in xr:
            for y in yr:
                for z in zr:
                    X.append(x); Y.append(y); Z.append(z)
        return X, Y, Z

    for c in S["cajas"]:
        if c[0] == "box":
            X, Y, Z = caja_xyz(c[1], c[2], c[3])
            col, op = c[4], (0.18 if c[4] == C_CONC else 0.35)
        else:
            poly, (y0, y1), col = c[1], c[2], c[3]
            X = [p[0] for p in poly] * 2
            Z = [p[1] for p in poly] * 2
            Y = [y0] * len(poly) + [y1] * len(poly)
            op = 0.22
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, alphahull=0, color=col, opacity=op, hoverinfo="skip",
                                showlegend=False, flatshading=True))

    def lineas(grupo, color, ancho, nombre, dash=None):
        X, Y, Z = [], [], []
        for pl in grupo:
            for p in pl:
                X.append(p[0]); Y.append(p[1]); Z.append(p[2])
            X.append(None); Y.append(None); Z.append(None)
        if X:
            fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name=nombre, hoverinfo="name",
                                       line=dict(color=color, width=ancho, dash=dash)))

    G = S["G"]
    lineas(S["long"], C_LONG, 7, "Barras longitudinales" + (" (ref.)" if G.get("ref") and G["tipo"] != "mensula" else "")
           if G["tipo"] != "mensula" else "Tensor y barras de la columna")
    lineas(S["est_c"], AZ_EST, 4, "Estribos por cálculo" if G["tipo"] != "mensula" else "Estribos Ah")
    lineas(S["est_m"], AZ_MONT, 4, "Estribos de montaje")
    lineas(S["otros"], "#7B241C", 9, "Barra transversal soldada")
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0) if G["tipo"] == "mensula" else
                               dict(x=0.45, y=-3.1, z=1.3) if G["tipo"] in ("viga", "tramo") else dict(x=1.6, y=-1.6, z=0.6))),
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        legend=dict(orientation="h", y=1.02, x=0))
    return fig


def figura_3d_mpl(D, R, rec=4.0):
    """Vista 3D estática con matplotlib (si plotly no está instalado)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = _segmentos_3d(D, R, rec)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    for c in S["cajas"]:
        if c[0] == "box":
            (x0, x1), (y0, y1), (z0, z1) = c[1], c[2], c[3]
            for xs, ys, zs in (([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
                               ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
                               ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
                               ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
                               ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
                               ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1])):
                ax.plot(xs, ys, zs, color="#999", lw=0.6)
        else:
            poly, (y0, y1) = c[1], c[2]
            for y in (y0, y1):
                ax.plot([p[0] for p in poly + poly[:1]], [y] * (len(poly) + 1), [p[1] for p in poly + poly[:1]],
                        color="#999", lw=0.6)
    for grupo, colr, lw in ((S["long"], C_LONG, 2), (S["est_c"], AZ_EST, 1), (S["est_m"], AZ_MONT, 1),
                            (S["otros"], "#7B241C", 3)):
        for pl in grupo:
            ax.plot([p[0] for p in pl], [p[1] for p in pl], [p[2] for p in pl], color=colr, lw=lw)
    ax.set_box_aspect(None)
    try:
        xs = [p[0] for g in (S["long"],) for pl in g for p in pl]
        ys = [p[1] for g in (S["long"],) for pl in g for p in pl]
        zs = [p[2] for g in (S["long"],) for pl in g for p in pl]
        ax.set_box_aspect((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1))
    except Exception:
        pass
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- visor 3D realista (three.js)
def _zonas_estribos(D, R, G):
    """Agrupa los estribos por zona de diseño: [(nombre, color, [x], etiqueta, (x_ini, x_fin))]."""
    PAL = ["#FF1E1E", "#FF8C00", "#00C9A7", "#B44CFF", "#FFD000"]
    L = G["L"]
    zonas = []
    if G["tipo"] == "columna":
        cc = R.conf_col
        xe = 5 + cc["n0"] * cc["s0"]
        inf = [x for x, _ in G["est"] if x <= xe + 1e-6]
        sup = [x for x, _ in G["est"] if x >= L - xe - 1e-6]
        cen = [x for x, _ in G["est"] if xe + 1e-6 < x < L - xe - 1e-6]
        zonas.append((f"Zona de confinamiento Lo = {cc['Lo']:.0f} cm", PAL[0], inf + sup,
                      f"Lo = {cc['Lo']:.0f} cm: 1 @ 5 + {cc['n0']} @ {cc['s0']:g} cm", (0, cc["Lo"])))
        zonas.append((f"Zona central @ {cc['sc']:g} cm", PAL[2], cen, f"Resto @ {cc['sc']:g} cm", (cc["Lo"], L - cc["Lo"])))
        return zonas
    if G["tipo"] == "tramo":
        s = R.tramos[0][1]
        return [(f"Estribos @ {s:g} cm", PAL[0], [x for x, _ in G["est"]], f"□ Ø{D.estribo} @ {s:g} cm", (0, L))]
    # viga: límites de cada tramo desde la cara (mitad izquierda; la derecha es simétrica)
    lim, x = [], R.primero
    for n, s in R.tramos:
        if n == "resto":
            lim.append((x, L / 2, "resto", s))
            break
        lim.append((x, x + n * s, n, s))
        x += n * s
    asig = {i: [] for i in range(len(lim))}
    mont = []
    for xp, t in G["est"]:
        xl = min(xp, L - xp)
        if t == "montaje":
            mont.append(xp)
            continue
        for i, (a, b, n, s) in enumerate(lim):
            if xl <= b + 1e-6 or i == len(lim) - 1:
                asig[i].append(xp)
                break
    for i, (a, b, n, s) in enumerate(lim):
        cnt = (n if n != "resto" else len([xp for xp in asig[i] if xp <= L / 2 + 1e-6]))
        conf = (i == 0 and getattr(R, "zona_conf", 0))
        nom = ((f"Zona de confinamiento 2h = {R.zona_conf:g} cm" if conf else ("Resto" if n == "resto" else f"Tramo {i + 1}"))
               + ": " + (f"1 @ {R.primero:g} + " if i == 0 else "") + f"{cnt} @ {s:g} cm")
        et = (f"2h = {R.zona_conf:g} cm · " if conf else "") + (f"1 @ {R.primero:g} + " if i == 0 else "") + \
             (f"{n} @ {s:g} cm" if n != "resto" else f"resto @ {s:g} cm")
        zonas.append((nom, PAL[i % len(PAL)], asig[i], et, (0 if i == 0 else a, b)))
    if mont:
        zonas.append((f"Montaje @ {R.smax:g} cm (Vu ≤ φVc/2)", "#1E90FF", mont, f"montaje @ {R.smax:g} cm",
                      (lim[-1][1], L / 2)))
    return zonas


def visor_3d_html(D, R, rec=4.0, alto=640):
    """Visor three.js: barras corrugadas con volumen, colores por zona de diseño, etiquetas con los
    resultados, cotas 3D y, en la ménsula, la biela y las fuerzas. Se mueve con el mouse."""
    import json
    S = _segmentos_3d(D, R, rec)
    G = S["G"]
    P = lambda p: [round(p[0], 2), round(p[2], 2), round(p[1], 2)]       # (x, y, z) -> three.js (x, z, y)
    gk = gancho(D)[0]
    r_est = G["dbe"] / 2
    grupos, etiq, cotas, flechas, bielas = [], [], [], [], []
    m = R.metr
    if G["tipo"] == "mensula":
        h, d, bw, lm, a = D.h, D.d, D.bw, D.lm_proy, D.a_carga
        db_p = BARRAS[D.barra_ppal][0]
        nb = m["nb"]
        ten = S["long"][:nb]
        colb = S["long"][nb:]
        grupos += [dict(nombre=f"Tensor: {nb} Ø{D.barra_ppal} (T = {R.biela['T']:.1f} t)", r=db_p / 2, color="#FF1E1E",
                        doblez=4 * db_p / 2, lineas=[[P(p) for p in pl] for pl in ten]),
                   dict(nombre=f"Estribos horizontales Ah: {m['ne']} □Ø{D.estribo} @ {m['se']:g} cm", r=r_est,
                        color="#00C9A7", doblez=5 * r_est, lineas=[[P(p) for p in pl] for pl in S["est_c"]]),
                   dict(nombre="Barra transversal soldada", r=db_p / 2 * 1.05, color="#7B241C", doblez=0,
                        lineas=[[P(p) for p in pl] for pl in S["otros"]]),
                   dict(nombre="Barras de la columna (referenciales)", r=0.8, color="#7F8C8D", doblez=0,
                        lineas=[[P(p) for p in pl] for pl in colb])]
        st_ = getattr(R, "stm", {}) or {}
        z = st_.get("z", R.biela["z"])
        ws = st_.get("ws", 10)
        bielas.append(dict(a=P((a, bw / 2, d)), b=P((0.0, bw / 2, d - z)), r=max(ws / 2, 3), w=bw * 0.8,
                           color="#2F80FF", nombre=f"Biela (puntal) de concreto: C = {R.biela['C']:.1f} t"))
        flechas += [dict(o=P((a, bw / 2, h + 45)), d=[0, -1, 0], l=40, c="#D62828"),
                    dict(o=P((a - 22, bw / 2, h + 1)), d=[1, 0, 0], l=20, c="#F77F00")]
        ldh = m["ldh"]
        etiq += [dict(p=P((a, bw / 2, h + 48)), t=f"Vu = {D.Vu_sec:g} t", c="#D62828"),
                 dict(p=P((a - 26, bw / 2, h + 3)), t=f"Nuc = {R.biela['Nuc']:.1f} t", c="#F77F00"),
                 dict(p=P((lm * 0.55, bw + 2, d + 4)), t=f"Tensor {nb} Ø{D.barra_ppal} · T = {R.biela['T']:.1f} t",
                      c="#D62828"),
                 dict(p=P((a * 0.2, bw / 2, d - z * 0.8)), t=f"Biela C = {R.biela['C']:.1f} t · θ = "
                      f"{math.degrees(R.biela['th']):.0f}°", c="#2F6DB5"),
                 dict(p=P((lm + 6, bw + 2, d - m["se"] - 4)), t=f"Ah: {m['ne']} □Ø{D.estribo} @ {m['se']:g} cm",
                      c="#2A9D8F"),
                 dict(p=P((lm - rec, bw + 2, d)), t="Barra transversal soldada", c="#7B241C"),
                 dict(p=P((-min(ldh, D.col_b - rec) / 2, bw + 2, d + 6)), t=f"Gancho 90° · ldh = {ldh:.0f} cm",
                      c="#6D597A")]
        etiq.append(dict(p=P((a / 2, -6, h + 6)), t=f"a = {a:g} cm", c="#444"))
        cotas += [dict(a=P((0, -6, h + 4)), b=P((a, -6, h + 4)), t="", c="#444"),
                  dict(a=P((-min(ldh, D.col_b - rec), -6, d)), b=P((0, -6, d)), t="", c="#6D597A")]
        titulo = "Ménsula corta: biela y tensor"
    else:
        poli = _poli_estribo(G, gk)
        col = G.get("vertical", False)
        Pm = (lambda s, u, v: P((u, v, s))) if col else (lambda s, u, v: P((s, u, v)))
        L, bw, h = G["L"], G["bw"], G["h"]
        grupos.append(dict(nombre="Barras longitudinales" + (" (referenciales)" if G.get("ref") else ""),
                           r=G["db"] / 2, color="#4A5568", doblez=0, lineas=[[P(p) for p in pl] for pl in S["long"]]))
        for iz, (nom, colr, xs, et, (x0, x1)) in enumerate(_zonas_estribos(D, R, G)):
            grupos.append(dict(nombre=nom, r=r_est, color=colr, doblez=5 * r_est,
                               lineas=[[Pm(x, u, v) for (u, v) in poli] for x in xs]))
            if col:
                for (a_, b_) in ((x0, x1),) if x1 - x0 < L - 1 else ((x0, x1),):
                    cotas.append(dict(a=Pm(a_, bw + 14, h / 2), b=Pm(b_, bw + 14, h / 2), t="", c=colr))
                etiq.append(dict(p=Pm((x0 + x1) / 2, bw + 18, h / 2), t=et, c=colr, z=True))
                if "Lo" in nom:
                    cotas.append(dict(a=Pm(L - x1, bw + 14, h / 2), b=Pm(L, bw + 14, h / 2), t="", c=colr))
                    etiq.append(dict(p=Pm(L - x1 / 2, bw + 18, h / 2), t=et, c=colr, z=True))
            else:
                yb = -0.32 * h - 6
                cotas.append(dict(a=Pm(x0, -4, yb), b=Pm(x1, -4, yb), t="", c=colr))
                etiq.append(dict(p=Pm((x0 + x1) / 2, -4, yb - 4 - (iz % 2) * 0.5 * h), t=et, c=colr, z=True))
                if G["tipo"] == "viga" and x1 <= L / 2 + 1e-6:
                    cotas.append(dict(a=Pm(L - x1, -4, yb), b=Pm(L - x0, -4, yb), t="", c=colr))
        if not col:
            cotas.append(dict(a=Pm(0, -4, -0.32 * h - 26), b=Pm(L, -4, -0.32 * h - 26),
                              t="", c="#222"))
            etiq.append(dict(p=Pm(L / 2, -4, -0.32 * h - 30),
                             t=f"{'L' if D.modo == 1 else ('ln' if D.modo == 2 else 'tramo')} = {L / 100:.2f} m", c="#222"))
            etiq.append(dict(p=Pm(L + (6 if G["tipo"] == "viga" else 2), bw + 4, h), t=f"Sección {bw:g} × {h:g} cm",
                             c="#222"))
            etiq.append(dict(p=Pm(R.primero if G["tipo"] == "viga" else 0, bw + 3, h + 8),
                             t=f"Primer estribo a {R.primero:g} cm de la cara" if G["tipo"] == "viga" else
                             f"□ Ø{D.estribo} · gancho 135° de {gk:.1f} cm", c="#D62828"))
            if G["tipo"] == "viga":
                etiq.append(dict(p=Pm(L / 2, bw / 2, h + 14),
                                 t=f"Vu(d) = {m['Vud']:.1f} t · φVc = {m['phiVc']:.1f} t · s máx = {m['smax']:g} cm",
                                 c="#222"))
        else:
            cotas.append(dict(a=Pm(0, -16, h / 2), b=Pm(L, -16, h / 2), t="", c="#222"))
            etiq.append(dict(p=Pm(L / 2, -20, h / 2), t=(f"ln = {L / 100:.2f} m" if G["tipo"] == "columna"
                                                         else f"tramo dibujado = {L / 100:.2f} m"), c="#222"))
            etiq.append(dict(p=Pm(L + 8, bw, h), t=f"Sección {bw:g} × {h:g} cm · gancho 135° de {gk:.1f} cm", c="#222"))
        titulo = {"viga": "Viga", "tramo": "Sección con carga axial", "columna": "Columna"}[G["tipo"]]
    solidos = []
    for c in S["cajas"]:
        if c[0] == "box":
            (x0, x1), (y0, y1), (z0, z1) = c[1], c[2], c[3]
            solidos.append(dict(t="box", min=P((x0, y0, z0)), max=P((x1, y1, z1)),
                                rol="elemento" if c[4] == C_CONC else ("placa" if c[4] == "#555" else "apoyo")))
        else:
            poly, (y0, y1) = c[1], c[2]
            solidos.append(dict(t="prisma", poly=[[round(p[0], 2), round(p[1], 2)] for p in poly], z0=y0, z1=y1,
                                rol="elemento"))
    datos = dict(grupos=grupos, solidos=solidos, etiq=etiq, cotas=cotas, flechas=flechas, bielas=bielas,
                 titulo=titulo, tipo=G["tipo"], dist=R.distribucion)
    return (_VISOR_HTML.replace("__DATOS__", json.dumps(datos, ensure_ascii=False))
            .replace("__ALTO__", str(alto)))


_VISOR_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
 html,body{margin:0;height:100%;background:#eef1f5;font-family:Segoe UI,Arial,sans-serif;overflow:hidden}
 #v{width:100%;height:__ALTO__px;position:relative;overflow:hidden}
 #hud{position:absolute;left:10px;top:10px;background:rgba(255,255,255,.94);border-radius:8px;padding:8px 10px;
      font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.15);max-width:340px;z-index:3}
 #hud b{font-size:13px}
 #hud label{display:block;margin-top:3px;cursor:pointer}
 #btns{position:absolute;right:10px;top:10px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;z-index:3}
 #btns button{border:1px solid #c6cdd6;background:#fff;border-radius:6px;padding:5px 9px;font-size:12px;cursor:pointer}
 #btns button:hover{background:#e8eef7}
 #ley{position:absolute;right:10px;top:50px;max-height:calc(100% - 70px);width:300px;overflow:auto;background:rgba(255,255,255,.95);
      border-radius:8px;padding:8px 12px;font-size:12px;z-index:3;line-height:1.45;box-shadow:0 1px 4px rgba(0,0,0,.14)}
 #ley h4{margin:6px 0 4px;font-size:12.5px;color:#0F2A52;border-bottom:1px solid #dde3ea;padding-bottom:3px}
 #ley .it{display:flex;align-items:flex-start;gap:7px;margin:4px 0}
 #ley .sw{flex:0 0 18px;height:8px;border-radius:4px;margin-top:5px}
 #ayuda{position:absolute;left:12px;bottom:8px;font-size:11px;color:#556;background:rgba(255,255,255,.85);padding:4px 8px;border-radius:6px;z-index:3}
 #pad{position:absolute;left:12px;bottom:34px;display:grid;gap:4px;z-index:3;
      grid-template-areas:". u . zi" "l c r ." ". d . zo";grid-template-columns:repeat(4,34px)}
 #pad button{height:30px;border:1px solid #c6cdd6;background:rgba(255,255,255,.95);border-radius:6px;font-size:14px;cursor:pointer}
 #pad button:hover{background:#e8eef7}
 #err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#a33;font-size:14px;text-align:center}
 .lab{position:absolute;transform:translate(-50%,-100%);white-space:nowrap;font-size:11.5px;font-weight:600;
      background:rgba(255,255,255,.92);border-radius:5px;padding:2px 7px;border-left:4px solid #333;pointer-events:none;
      box-shadow:0 1px 3px rgba(0,0,0,.18);z-index:2}
</style></head><body><div id="v">
<div id="hud"><b id="tit"></b><div id="dist" style="margin:3px 0 4px;color:#333"></div>
 <label><input type="checkbox" id="cLab" checked> Cotas de cada zona</label>
 <label><input type="checkbox" id="cCol" checked> Colores por zona de diseño</label>
 <label><input type="checkbox" id="cConc" checked> Mostrar concreto</label>
 <label>Transparencia <input type="range" id="op" min="0" max="97" value="91" style="width:120px;vertical-align:middle"></label>
 <label>Corte longitudinal <input type="range" id="cut" min="0" max="100" value="100" style="width:120px;vertical-align:middle"></label></div>
<div id="btns"><button data-v="iso">3D</button><button data-v="fre">Frontal</button><button data-v="lat">Lateral</button>
 <button data-v="sup">Superior</button><button id="giro">Girar</button><button id="full">⛶ Pantalla completa</button></div>
<div id="pad"><button data-p="u" style="grid-area:u">▲</button><button data-p="l" style="grid-area:l">◀</button>
 <button data-p="c" style="grid-area:c">⟲</button><button data-p="r" style="grid-area:r">▶</button>
 <button data-p="d" style="grid-area:d">▼</button><button data-p="zi" style="grid-area:zi">＋</button>
 <button data-p="zo" style="grid-area:zo">－</button></div>
<div id="ley"></div>
<div id="ayuda">Izq.: girar · Der. o flechas: mover · Rueda o +/−: zoom</div>
<div id="err">No se pudo cargar el visor 3D (requiere conexión a internet para three.js).</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATOS__;
if (typeof THREE === "undefined") { document.getElementById("err").style.display = "flex"; }
else {
const cont = document.getElementById("v");
const W = () => cont.clientWidth, H = () => cont.clientHeight;
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W(), H());
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = true;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.NoToneMapping;
cont.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef1f5);

function texConcreto() {
  const c = document.createElement("canvas"); c.width = c.height = 256; const g = c.getContext("2d");
  g.fillStyle = "#b9bcbf"; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 9000; i++) { const v = 150 + Math.random() * 70 | 0, a = Math.random() * .35;
    g.fillStyle = `rgba(${v},${v},${v - 4},${a})`; g.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 2, 1 + Math.random() * 2); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.encoding = THREE.sRGBEncoding; return t;
}
function texCorrugado() {
  const c = document.createElement("canvas"); c.width = 64; c.height = 16; const g = c.getContext("2d");
  const gr = g.createLinearGradient(0, 0, 64, 0);
  gr.addColorStop(0, "#000"); gr.addColorStop(.42, "#000"); gr.addColorStop(.5, "#fff"); gr.addColorStop(.58, "#000"); gr.addColorStop(1, "#000");
  g.fillStyle = gr; g.fillRect(0, 0, 64, 16);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
}
const TC = texConcreto(), TB = texCorrugado();
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 0.8));
const sol = new THREE.DirectionalLight(0xffffff, 0.9); scene.add(sol);
sol.castShadow = true; sol.shadow.mapSize.set(2048, 2048); sol.shadow.bias = -0.0005;

let bb = new THREE.Box3();
DATA.solidos.forEach(s => { if (s.t === "box") { bb.expandByPoint(new THREE.Vector3(...s.min)); bb.expandByPoint(new THREE.Vector3(...s.max)); }
  else s.poly.forEach(p => { bb.expandByPoint(new THREE.Vector3(p[0], p[1], s.z0)); bb.expandByPoint(new THREE.Vector3(p[0], p[1], s.z1)); }); });
DATA.grupos.forEach(g => g.lineas.forEach(l => l.forEach(p => bb.expandByPoint(new THREE.Vector3(...p)))));
const C = bb.getCenter(new THREE.Vector3()), SZ = bb.getSize(new THREE.Vector3()), R0 = SZ.length();
const pisoY = bb.min.y - Math.max(8, SZ.y * .5);
const piso = new THREE.Mesh(new THREE.PlaneGeometry(R0 * 4, R0 * 4), new THREE.ShadowMaterial({opacity: .16}));
piso.rotation.x = -Math.PI / 2; piso.position.set(C.x, pisoY, C.z); piso.receiveShadow = true; scene.add(piso);
const grid = new THREE.GridHelper(R0 * 3, 30, 0xc5cbd3, 0xdde2e8); grid.position.set(C.x, pisoY + .1, C.z); scene.add(grid);
sol.position.set(C.x + R0 * .6, C.y + R0 * 1.1, C.z + R0 * .8); sol.target.position.copy(C); scene.add(sol.target);
const sc = sol.shadow.camera; sc.left = sc.bottom = -R0; sc.right = sc.top = R0; sc.near = 1; sc.far = R0 * 4; sc.updateProjectionMatrix();

const eje = SZ.x >= SZ.y && SZ.x >= SZ.z ? "x" : (SZ.y >= SZ.z ? "y" : "z");
const nrm = {x: new THREE.Vector3(-1, 0, 0), y: new THREE.Vector3(0, -1, 0), z: new THREE.Vector3(0, 0, -1)}[eje];
const corte = new THREE.Plane(nrm, bb.max[eje] + 1);

const matConc = new THREE.MeshStandardMaterial({map: TC, color: 0xd8dadc, roughness: .95, metalness: 0,
  transparent: true, opacity: .09, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: [corte]});
const matApoyo = new THREE.MeshStandardMaterial({map: TC, color: 0xa9adb3, roughness: .95, transparent: true, opacity: .45,
  depthWrite: false, clippingPlanes: [corte]});
const matPlaca = new THREE.MeshStandardMaterial({color: 0x666b70, metalness: .7, roughness: .4});
const concreto = [];
DATA.solidos.forEach(s => {
  let geo;
  if (s.t === "box") {
    const a = new THREE.Vector3(...s.min), b = new THREE.Vector3(...s.max), d = b.clone().sub(a);
    geo = new THREE.BoxGeometry(Math.abs(d.x), Math.abs(d.y), Math.abs(d.z));
    geo.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
  } else {
    const sh = new THREE.Shape(s.poly.map(p => new THREE.Vector2(p[0], p[1])));
    geo = new THREE.ExtrudeGeometry(sh, {depth: s.z1 - s.z0, bevelEnabled: false}); geo.translate(0, 0, s.z0);
  }
  const m = new THREE.Mesh(geo, s.rol === "placa" ? matPlaca : (s.rol === "apoyo" ? matApoyo : matConc));
  m.receiveShadow = true; m.castShadow = s.rol === "placa"; m.renderOrder = 2; scene.add(m);
  if (s.rol !== "placa") { concreto.push(m);
    const ed = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color: 0x7d838b, transparent: true, opacity: .55, clippingPlanes: [corte]}));
    scene.add(ed); concreto.push(ed); }
});

function camino(pts, rd) {
  const P = pts.map(p => new THREE.Vector3(...p)), path = new THREE.CurvePath();
  if (P.length < 2) return null;
  let cur = P[0].clone();
  for (let i = 1; i < P.length; i++) {
    const a = P[i - 1], b = P[i], c = P[i + 1];
    if (c && rd > 0) {
      const u = a.clone().sub(b), w = c.clone().sub(b), lu = u.length(), lw = w.length();
      const r = Math.min(rd, lu * .45, lw * .45);
      const p1 = b.clone().add(u.normalize().multiplyScalar(r)), p2 = b.clone().add(w.normalize().multiplyScalar(r));
      if (cur.distanceTo(p1) > 1e-3) path.add(new THREE.LineCurve3(cur.clone(), p1));
      path.add(new THREE.QuadraticBezierCurve3(p1, b.clone(), p2)); cur = p2;
    } else { if (cur.distanceTo(b) > 1e-3) path.add(new THREE.LineCurve3(cur.clone(), b.clone())); cur = b.clone(); }
  }
  return path;
}
const ACERO = 0x5d5048;
const mats = [];
DATA.grupos.forEach(g => {
  const lin = c => new THREE.Color(c).convertSRGBToLinear();
  const mat = new THREE.MeshStandardMaterial({color: lin(g.color), metalness: .05, roughness: .45, bumpMap: TB, bumpScale: .6,
    emissive: lin(g.color), emissiveIntensity: .22});
  mat.userData = {tipo: g.color}; mats.push(mat);
  g.lineas.forEach(l => {
    const path = camino(l, g.doblez); if (!path) return;
    const len = path.getLength();
    const geo = new THREE.TubeGeometry(path, Math.max(8, Math.min(900, Math.round(len / 1.2))), g.r, 12, false);
    const uv = geo.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setX(i, uv.getX(i) * len / (g.r * 1.6));
    const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true; scene.add(m);
    [0, 1].forEach(t => { const p = path.getPointAt(t), tg = path.getTangentAt(t);
      const cap = new THREE.Mesh(new THREE.CircleGeometry(g.r, 14), mat);
      cap.position.copy(p); cap.lookAt(p.clone().add(tg.multiplyScalar(t ? 1 : -1))); scene.add(cap); });
  });
});

// ---------- biela (puntal) de concreto, flechas de fuerzas y cotas
const extras = [];
DATA.bielas.forEach(b => {
  const A = new THREE.Vector3(...b.a), B = new THREE.Vector3(...b.b), dir = B.clone().sub(A), len = dir.length();
  const geo = new THREE.CylinderGeometry(b.r, b.r * 1.35, len, 28, 1, false);
  geo.scale(1, 1, b.w / (2 * b.r));
  const mat = new THREE.MeshStandardMaterial({color: b.color, transparent: true, opacity: .55, depthWrite: false, roughness: .6, emissive: b.color, emissiveIntensity: .25});
  const m = new THREE.Mesh(geo, mat);
  m.position.copy(A.clone().add(B).multiplyScalar(.5));
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  scene.add(m); extras.push(m);
});
DATA.flechas.forEach(f => {
  const ar = new THREE.ArrowHelper(new THREE.Vector3(...f.d).normalize(), new THREE.Vector3(...f.o), f.l, f.c, f.l * .3, f.l * .16);
  ar.line.material.linewidth = 3; scene.add(ar); extras.push(ar);
});
const cotas3d = [];
DATA.cotas.forEach(c => {
  const A = new THREE.Vector3(...c.a), B = new THREE.Vector3(...c.b);
  const mat = new THREE.LineBasicMaterial({color: c.c});
  const dir = B.clone().sub(A); const tk = Math.max(2.5, dir.length() * .0) + 3;
  const up = new THREE.Vector3(0, 1, 0); if (Math.abs(dir.clone().normalize().dot(up)) > .9) up.set(1, 0, 0);
  const n = up.multiplyScalar(tk);
  const g = new THREE.BufferGeometry().setFromPoints([A, B, A.clone().add(n), A.clone().sub(n), B.clone().add(n), B.clone().sub(n)]);
  g.setIndex([0, 1, 2, 3, 4, 5]);
  const ln = new THREE.LineSegments(g, mat); scene.add(ln); cotas3d.push(ln);
});

// ---------- etiquetas HTML ancladas a puntos 3D
const labs = [];
function posLabs() {
  const vis = document.getElementById("cLab").checked;
  labs.forEach(l => { const v = l.p.clone().project(cam);
    const ok = vis && v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15;
    l.el.style.display = ok ? "block" : "none";
    if (ok) { l.el.style.left = ((v.x + 1) / 2 * W()) + "px"; l.el.style.top = ((1 - v.y) / 2 * H() - 4) + "px"; } });
}

const cam = new THREE.PerspectiveCamera(38, W() / H(), R0 / 500, R0 * 20);
const desplazar = () => {};
const ctl = new THREE.OrbitControls(cam, renderer.domElement);
ctl.enableDamping = true; ctl.dampingFactor = .08; ctl.target.copy(C); ctl.screenSpacePanning = true;
// encuadre automático: el elemento ocupa el mayor tamaño posible, centrado, sin quedar bajo los paneles
const esquinas = [];
[bb.min.x, bb.max.x].forEach(x => [bb.min.y, bb.max.y].forEach(y => [bb.min.z, bb.max.z].forEach(z => esquinas.push(new THREE.Vector3(x, y, z)))));
function obstaculos() {
  const r0 = cont.getBoundingClientRect();
  return ["hud", "ley", "pad"].map(id => document.getElementById(id)).filter(e => e && e.offsetParent !== null).map(e => {
    const r = e.getBoundingClientRect();
    return {x0: (r.left - r0.left) / W() * 2 - 1, x1: (r.right - r0.left) / W() * 2 - 1,
            y0: 1 - (r.bottom - r0.top) / H() * 2, y1: 1 - (r.top - r0.top) / H() * 2};
  });
}
function proyectar() { cam.updateMatrixWorld(); return esquinas.map(p => p.clone().project(cam)); }
function centrar(dir, d) {                       // ubica la cámara y desplaza para centrar el elemento en pantalla
  ctl.target.copy(C); cam.position.copy(C).add(dir.clone().multiplyScalar(d)); cam.lookAt(C);
  for (let k = 0; k < 3; k++) {
    const q = proyectar();
    const cx = (Math.min(...q.map(v => v.x)) + Math.max(...q.map(v => v.x))) / 2;
    const cy = (Math.min(...q.map(v => v.y)) + Math.max(...q.map(v => v.y))) / 2;
    const dist = cam.position.distanceTo(ctl.target), hh = dist * Math.tan(cam.fov * Math.PI / 360), ww = hh * cam.aspect;
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0), up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
    const v = right.multiplyScalar(cx * ww).add(up.multiplyScalar(cy * hh));
    cam.position.add(v); ctl.target.add(v); cam.lookAt(ctl.target);
  }
}
function cabe(dir, d, obs) {
  centrar(dir, d);
  const q = proyectar();
  return q.every(v => v.z < 1 && Math.abs(v.x) < 0.97 && Math.abs(v.y) < 0.93 &&
    !obs.some(o => v.x > o.x0 && v.x < o.x1 && v.y > o.y0 && v.y < o.y1));
}
function encuadrar(dir) {
  cam.aspect = W() / H(); cam.clearViewOffset(); cam.updateProjectionMatrix();
  const obs = obstaculos();
  let lo = R0 * 0.1, hi = R0 * 8;
  for (let i = 0; i < 36; i++) { const mid = (lo + hi) / 2; if (cabe(dir, mid, obs)) hi = mid; else lo = mid; }
  if (!cabe(dir, hi, obs)) {                     // si los paneles no dejan espacio, encuadra solo por el borde
    lo = R0 * 0.1; hi = R0 * 8;
    for (let i = 0; i < 36; i++) { const mid = (lo + hi) / 2; if (cabe(dir, mid, [])) hi = mid; else lo = mid; }
  }
  return hi;
}
function vista(v) {
  const dirs0 = {iso: [.45, .5, .85], fre: [0, 0, 1], lat: [1, 0, 0], sup: [0, 1, .0001]};
  if (eje === "y" && DATA.tipo !== "mensula") dirs0.iso = [.8, .35, .9];
  cam.up.set(0, 1, 0);
  const dir = new THREE.Vector3(...dirs0[v]).normalize();
  const d = encuadrar(dir);
  centrar(dir, d * 1.03); ctl.update();
}
document.querySelectorAll("#btns button[data-v]").forEach(b => b.onclick = () => { ultimaVista = b.dataset.v; vista(b.dataset.v); });
function mover(dx, dy) {                       // desplaza cámara y objetivo en el plano de la pantalla
  const k = cam.position.distanceTo(ctl.target) * .12;
  const right = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(cam.matrix, 1);
  const v = right.multiplyScalar(dx * k).add(up.multiplyScalar(dy * k));
  cam.position.add(v); ctl.target.add(v); ctl.update();
}
function zoom(f) { const v = cam.position.clone().sub(ctl.target).multiplyScalar(f); cam.position.copy(ctl.target).add(v); ctl.update(); }
document.querySelectorAll("#pad button").forEach(b => b.onclick = () => ({u: () => mover(0, 1), d: () => mover(0, -1),
  l: () => mover(-1, 0), r: () => mover(1, 0), zi: () => zoom(.8), zo: () => zoom(1.25), c: () => vista("iso")})[b.dataset.p]());
window.addEventListener("keydown", e => { const m = {ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0]}[e.key];
  if (m) { mover(...m); e.preventDefault(); } if (e.key === "+") zoom(.8); if (e.key === "-") zoom(1.25); });
document.getElementById("full").onclick = () => { const el = document.getElementById("v");
  if (!document.fullscreenElement) (el.requestFullscreen || el.webkitRequestFullscreen).call(el); else document.exitFullscreen(); };
document.addEventListener("fullscreenchange", () => { document.getElementById("v").style.height = document.fullscreenElement ? "100vh" : "__ALTO__px";
  setTimeout(() => { cam.aspect = W() / H(); desplazar(); cam.updateProjectionMatrix(); renderer.setSize(W(), H()); }, 60); });
document.getElementById("giro").onclick = function () { ctl.autoRotate = !ctl.autoRotate; this.style.background = ctl.autoRotate ? "#dbe7f7" : "#fff"; };

document.getElementById("tit").textContent = DATA.titulo;
document.getElementById("dist").textContent = DATA.dist;
document.getElementById("cConc").onchange = e => concreto.forEach(m => m.visible = e.target.checked);
document.getElementById("op").oninput = e => { const t = 1 - e.target.value / 100; matConc.opacity = t; matApoyo.opacity = Math.min(1, t + .25); };
document.getElementById("cut").oninput = e => { const f = e.target.value / 100; corte.constant = bb.min[eje] + f * (bb.max[eje] - bb.min[eje]) + (f >= 1 ? 1 : 0); };
document.getElementById("cLab").onchange = e => cotas3d.forEach(c => c.visible = e.target.checked);
function colores() {
  const on = document.getElementById("cCol").checked;
  mats.forEach(m => { m.color.set(on ? m.userData.tipo : ACERO).convertSRGBToLinear(); m.emissive.set(on ? m.userData.tipo : 0x000000).convertSRGBToLinear();
    m.metalness = on ? .05 : .55; });
  const it = (c, t, op) => `<div class="it"><span class="sw" style="background:${c};opacity:${op || 1}"></span><span>${t}</span></div>`;
  let h = "<h4>Qué es cada color</h4>";
  DATA.grupos.forEach(g => h += it(on ? g.color : "#5d5048", g.nombre));
  DATA.bielas.forEach(b => h += it(b.color, b.nombre, .6));
  h += "<h4>Resultados</h4>";
  const vistos = new Set();
  DATA.etiq.forEach(e => { if (e.z || vistos.has(e.t)) return; vistos.add(e.t); h += it(e.c, e.t); });
  document.getElementById("ley").innerHTML = h;
}
document.getElementById("cCol").onchange = colores; colores();
vista("iso");

let ultimaVista = "iso";
window.addEventListener("resize", () => { cam.aspect = W() / H(); desplazar(); cam.updateProjectionMatrix(); renderer.setSize(W(), H()); vista(ultimaVista); });
(function loop() { requestAnimationFrame(loop); ctl.update(); renderer.render(scene, cam); posLabs(); })();
}
</script></body></html>"""


# ===========================================================================
# INTERFAZ STREAMLIT
# ===========================================================================
def app():
    import io
    import streamlit as st

    st.set_page_config(page_title="Diseño a corte E.060", page_icon="🏗️", layout="wide")
    st.title("Diseño a corte de vigas de concreto armado")
    st.caption("Norma E.060 · Procedimiento de Ortega (2014), cap. 7 · Concreto Armado I, UNSCH")

    with st.sidebar:
        st.header("¿Qué desea hacer?")
        try:
            ini = min(max(int(st.query_params.get("ej", 0)), 0), len(EJEMPLOS) - 1)
        except ValueError:
            ini = 0
        accion = st.radio("Opción", ["✏️ Resolver un problema", "📘 Ver ejemplos del libro"],
                          index=1 if "ej" in st.query_params else 0, label_visibility="collapsed")
        if accion.startswith("✏️"):
            st.subheader("Tipo de problema")
            modo = st.radio("Tipo de problema", list(PLANTILLAS), label_visibility="collapsed",
                            format_func=lambda m: f"{PLANTILLAS[m]['icono']} {PLANTILLAS[m]['nombre']}")
            P = PLANTILLAS[modo]
            st.info(P["desc"])
            nombre = f"Problema nuevo: {P['nombre']}"
            ej = dict(datos=P["datos"], fuente="", enunciado="", libro=[], notas=[])
            k = f"nuevo{modo}"
        else:
            st.subheader("Ejemplo")
            nombre = st.selectbox("Ejemplo", list(EJEMPLOS), index=ini, label_visibility="collapsed")
            ej = EJEMPLOS[nombre]
            modo = ej["datos"].modo
            k = nombre[:12]
            st.caption(f"Tipo: {MODOS[modo]}")
        D0 = ej["datos"]
        st.caption("Ingrese los datos sobre los dibujos y pulse **Calcular**.")
        with st.expander("Criterios de distribución"):
            red = st.number_input("Redondeo de s (cm)", 0.5, 10.0, float(5.0 if modo == 2 else D0.redondeo),
                                  step=0.5, key=k + f"red{modo}")
            paso = st.number_input("Cambio de tramo cuando s sube (cm)", 1.0, 10.0, float(D0.paso), key=k + "paso")
            mine = st.number_input("Estribos mínimos por tramo", 1, 10, int(D0.min_estribos), key=k + "min")

    # ---------------- Datos sobre los dibujos ----------------
    import streamlit.components.v1 as components
    caja_enunciado = st.empty()
    st.subheader(f"Datos del problema · {MODOS[modo]}")
    with st.form(k + f"form{modo}"):
        izq, der = st.columns([2.3, 1])
        D = Datos(modo=modo, titulo=D0.titulo, redondeo=red, paso=paso, min_estribos=int(mine))
        with izq:
            if modo == 1:
                a1, a2, a3 = st.columns([1, 1.2, 1])
                D.wu = a2.number_input("Carga última wu (t/m)", 0.1, 200.0, float(D0.wu), key=k + "wu")
            elif modo == 2:
                a1, a2 = st.columns(2)
                D.CM = a1.number_input("Carga muerta CM (t/m)", 0.0, 100.0, float(D0.CM), key=k + "cm")
                D.CV = a2.number_input("Carga viva CV (t/m)", 0.0, 100.0, float(D0.CV), key=k + "cv")
            elif modo == 4:
                a1, a2 = st.columns(2)
                D.Nu = a1.number_input("Nu (t): + compresión, − tracción", -500.0, 2000.0, float(D0.Nu), key=k + "nu")
                D.Vu_sec = a2.number_input("Cortante Vu (t)", 0.1, 500.0, float(D0.Vu_sec), key=k + "vus")
            elif modo == 5:
                a1, a2 = st.columns(2)
                D.Vu_sec = a1.number_input("Carga última Vu (t)", 0.1, 500.0, float(D0.Vu_sec), key=k + "vus")
                D.Nuc = a2.number_input("Fuerza horizontal Nuc (t), mín. 0.2·Vu", 0.0, 500.0, float(D0.Nuc), key=k + "nuc")
            fig_slot = st.empty()
            if modo == 1:
                b1, b2, b3 = st.columns([1, 1.2, 1])
                D.L = b2.number_input("Luz L (m)", 1.0, 30.0, float(D0.L), key=k + "L")
            elif modo == 2:
                b1, b2, b3 = st.columns(3)
                D.As_sup = b1.number_input("As⁻ en el extremo (cm²)", 0.0, 100.0, float(D0.As_sup), key=k + "as1")
                D.ln = b2.number_input("Luz libre ln (m)", 1.0, 20.0, float(D0.ln), key=k + "ln")
                D.As_inf = b3.number_input("As⁺ en el extremo (cm²)", 0.0, 100.0, float(D0.As_inf), key=k + "as2")
                D.db_long = b1.number_input("Menor Ø longitudinal (cm)", 0.9, 3.6, float(D0.db_long), key=k + "db")
            elif modo == 3:
                b1, b2 = st.columns(2)
                D.Vu_cara = b1.number_input("Vu en la cara del apoyo (t)", 0.1, 500.0, float(D0.Vu_cara), key=k + "vuc")
                D.luz_mitad = b2.number_input("Distancia a Vu = 0 (m)", 0.5, 15.0, float(D0.luz_mitad), key=k + "lm")
            elif modo == 5:
                b1, b2 = st.columns(2)
                D.a_carga = b1.number_input("Distancia de la carga a la cara, a (cm)", 2.0, 300.0, float(D0.a_carga), key=k + "ac")
                D.lm_proy = b2.number_input("Proyección de la ménsula, lm (cm)", 5.0, 400.0, float(D0.lm_proy), key=k + "lmp")
                D.placa_l = b1.number_input("Placa de apoyo: largo lp (cm)", 2.0, 100.0, float(D0.placa_l), key=k + "pl")
                D.placa_b = b2.number_input("Placa de apoyo: ancho bp (cm)", 2.0, 200.0, float(D0.placa_b), key=k + "pb")
            else:
                b1, b2 = st.columns(2)
                D.s_exist = b1.number_input("s existente a verificar (cm), 0 = ninguno", 0.0, 100.0,
                                            float(D0.s_exist), key=k + "sex")
                D.corte_total = b2.checkbox("Si hay tracción: despreciar Vc (corte total)", value=D0.corte_total,
                                            key=k + "ct")
                D.ln_col = b1.number_input("Altura libre de columna ln (m), 0 = no es columna", 0.0, 15.0,
                                           float(D0.ln_col), key=k + "lncol",
                                           help="Si es columna, se calcula la zona de confinamiento Lo (E.060 21.4.5)")
        with der:
            sec_slot = st.empty()
            c1, c2 = st.columns(2)
            D.bw = c1.number_input("bw (cm)", 10.0, 200.0, float(D0.bw), key=k + "bw")
            D.h = c2.number_input("h (cm)", 15.0, 300.0, float(D0.h), key=k + "h")
            D.d = c1.number_input("d (cm)", 10.0, 300.0, float(D0.d), key=k + "d")
            D.estribo = c2.selectbox("Estribo Ø", list(BARRAS), index=list(BARRAS).index(D0.estribo), key=k + "est")
            if modo == 5:
                D.h_borde = c1.number_input("h en el borde, hb (cm)", 5.0, 300.0, float(D0.h_borde), key=k + "hb")
                D.barra_ppal = c2.selectbox("Barras del tensor", list(BARRAS),
                                            index=list(BARRAS).index(D0.barra_ppal), key=k + "bp")
                D.col_b = c1.number_input("Columna: dimensión b (cm)", 15.0, 200.0, float(D0.col_b), key=k + "colb",
                                          help="Dimensión de la columna en la dirección del tensor (para el anclaje)")
            else:
                D.ramas = int(c1.number_input("Ramas", 2, 6, int(D0.ramas), key=k + "ram"))
                D.Av_manual = c2.number_input("Av manual (cm²)", 0.0, 20.0, float(D0.Av_manual), key=k + "av",
                                              help="0 = calcula ramas × área de la barra")
        st.markdown("**Materiales**")
        m1, m2, m3, m4, m5 = st.columns(5)
        D.fc = m1.number_input("f′c (kg/cm²)", 140.0, 700.0, float(D0.fc), key=k + "fc")
        D.fy = m2.number_input("fy estribo (kg/cm²)", 2800.0, 4200.0, float(D0.fy), step=1400.0, key=k + "fy")
        D.lam = m3.number_input("λ (concreto liviano)", 0.75, 1.0, float(D0.lam), step=0.05, key=k + "lam",
                                help="1.0 concreto normal; 0.85 arena liviana; 0.75 todo liviano")
        if modo in (1, 3, 4):
            D.db_long = m5.number_input("Ø barra longitudinal (cm)", 0.9, 3.6, float(D0.db_long), key=k + "dbl",
                                        help="Para comprobar d y el confinamiento")
        if modo not in (4, 5):
            D.Nu = m4.number_input("Carga axial Nu (t)", -500.0, 2000.0, float(D0.Nu), key=k + "nu",
                                   help="+ compresión, − tracción. Normalmente 0 en vigas.")
        with fig_slot.container():
            components.html('<body style="margin:0"><div style="height:290px">' + svg_elevacion(D) + '</div></body>', height=295)
        with sec_slot.container():
            components.html('<body style="margin:0"><div style="height:290px">' + svg_seccion(D) + '</div></body>', height=295)
        st.form_submit_button("🧮 Calcular", type="primary", use_container_width=True)

    with caja_enunciado.container(border=True):
        if ej["enunciado"]:
            st.markdown(f"**{nombre}**  \n{ej['enunciado']}  \n*Fuente: {ej['fuente']}*")
            enunciado_final = ej["enunciado"]
        else:
            st.markdown("**✏️ Enunciado del problema**")
            enu_user = st.text_area(
                "Enunciado", "", key=k + "enu_top", height=110, label_visibility="collapsed",
                placeholder="Escriba o pegue aquí el enunciado de su problema. Si lo deja vacío, se usará: "
                            + enunciado_auto(D),
                help="Este texto aparece al inicio de la memoria de cálculo en PDF.")
            enunciado_final = enu_user.strip() or enunciado_auto(D)
            if not enu_user.strip():
                st.caption("Enunciado automático con sus datos: " + enunciado_auto(D))

    R = calcular(D)
    m = R.metr
    c = st.columns(5)
    if modo == 5:
        c[0].metric("Vu", f"{D.Vu_sec:.2f} t")
        c[1].metric("a / d", f"{D.a_carga / D.d:.2f}")
        c[2].metric("Tensor T / biela C", f"{m.get('T', 0):.1f} / {m.get('C', 0):.1f} t")
        c[3].metric("Tensor a colocar", f"{m.get('As_fin', 0):.2f} cm²")
        c[4].metric("Ah", f"{m.get('Ah', 0):.2f} cm²")
    else:
        c[0].metric("Vu en la sección crítica", f"{m.get('Vud', 0):.2f} t")
        c[1].metric("φVc", f"{m.get('phiVc', 0):.2f} t")
        c[2].metric("Vs requerido", f"{m.get('Vs', 0):.2f} t")
        sv0 = m.get("s")
        c[3].metric("s por cálculo", "—" if not sv0 or sv0 > 900 else f"{sv0:.2f} cm")
        c[4].metric("s máx", f"{m.get('smax', 0):g} cm" if m.get("smax") else "—")
    sv = m.get("s")

    if R.ok:
        st.success(f"**Resultado:** {R.distribucion}")
    else:
        st.error("**Revisar:** " + (R.distribucion or "la sección no cumple"))
    for a in R.alertas:
        st.warning(a)

    t1, t2, t3, t4 = st.tabs(["📝 Paso a paso", "📈 Gráficos", "📐 Planos", "🧊 Vista 3D"])
    with t1:
        for titulo, lat, txt in R.pasos:
            st.markdown(f"**{titulo.strip()}**")
            for l, _ in lat:
                st.latex(l)
            if txt:
                st.caption(txt)
        st.markdown("**Referencias**")
        for ref in REFERENCIAS + ([REF_PERLES, REF_ACI] if modo == 5 else []):
            st.markdown(f"- {ref}")
    fig = None
    with t2:
        if R.tramos:
            try:
                lista = figuras(D, R)
                fig = lista[0][2]
                for i, (tit, desc, f_) in enumerate(lista, 1):
                    st.markdown(f"**Gráfico {i}. {tit}**")
                    st.pyplot(f_)
                    st.caption(desc)
                    if i > 1:
                        import matplotlib.pyplot as _plt
                        _plt.close(f_)
            except ImportError:
                st.error("Falta matplotlib. En la terminal: pip install matplotlib")
    ss = st.session_state
    info_pl = dict(proyecto=ss.get(k + "pr", "Concreto Armado I - UNSCH"),
                   elemento=ss.get(k + "el", D0.titulo if ej["enunciado"] else "Viga"),
                   elaborado=ss.get(k + "ela", ""), revisado=ss.get(k + "rev", "Ing. Taipe Carbajal, Javier"))
    rec_pl = float(ss.get(k + "rec", 4.0))
    with t3:
        if R.tramos:
            st.caption("Lámina A3 lista para imprimir: elevación acotada, cortes, detalle del estribo, cuadro de "
                       "estribos, metrado, especificaciones y cajetín. Los datos del cajetín se editan en "
                       "«Datos para la memoria de cálculo», más abajo.")
            fpl = plano_lamina(D, R, rec_pl, info_pl)
            st.pyplot(fpl)
            q1, q2 = st.columns(2)
            bpng, bpdf = io.BytesIO(), io.BytesIO()
            fpl.savefig(bpng, format="png", dpi=200)
            fpl.savefig(bpdf, format="pdf")
            import matplotlib.pyplot as _plt
            _plt.close(fpl)
            q1.download_button("⬇️ Plano en PNG", bpng.getvalue(), file_name="Plano_armado_corte.png",
                               use_container_width=True)
            q2.download_button("⬇️ Plano en PDF (A3)", bpdf.getvalue(), file_name="Plano_armado_corte.pdf",
                               mime="application/pdf", use_container_width=True)
    with t4:
        if R.tramos:
            html3d = visor_3d_html(D, R, rec_pl, alto=780)
            components.html(html3d, height=790)
            st.caption("Clic izquierdo y arrastrar: girar · clic derecho: mover · rueda: acercar. En el panel puede "
                       "ocultar el concreto, cambiar su transparencia, cortar el elemento a lo largo y colorear los "
                       "estribos por tipo. Necesita internet (carga three.js).")
            st.download_button("⬇️ Descargar modelo 3D (.html, se abre en el navegador)", html3d.encode("utf-8"),
                               file_name="Modelo_3D_armado.html", mime="text/html")
    # ---------------- Descargas: PDF y gráfico ----------------
    st.divider()
    with st.expander("📄 Datos para la memoria de cálculo (PDF)"):
        p1, p2, p3 = st.columns(3)
        info = dict(
            proyecto=p1.text_input("Proyecto / curso", "Concreto Armado I - UNSCH", key=k + "pr"),
            elemento=p2.text_input("Elemento", D0.titulo if ej["enunciado"] else "Viga", key=k + "el"),
            ubicacion=p3.text_input("Ubicación", "Ayacucho, Perú", key=k + "ub"),
            elaborado=p1.text_input("Elaborado por", "", key=k + "ela"),
            revisado=p2.text_input("Revisado por (docente / ingeniero)", "Ing. Taipe Carbajal, Javier", key=k + "rev"),
            rec=p3.number_input("Recubrimiento libre r (cm)", 2.0, 7.5, 4.0, step=0.5, key=k + "rec"),
            fuente=ej["fuente"],
        )
    info["enunciado"] = enunciado_final
    firma = repr(asdict(D)) + repr(info)
    b1, b2, b3 = st.columns(3)
    if b1.button("📄 Generar PDF", type="primary", disabled=not R.tramos, use_container_width=True):
        try:
            with st.spinner("Generando la memoria de cálculo..."):
                st.session_state["pdf"] = (firma, generar_pdf(D, R, info))
        except ImportError:
            st.error("Falta la librería reportlab. En la terminal escriba: pip install reportlab")
    guardado = st.session_state.get("pdf")
    if guardado and guardado[0] == firma:
        nombre_pdf = "Memoria_corte_" + "".join(ch if ch.isalnum() else "_" for ch in info["elemento"])[:40] + ".pdf"
        b2.download_button("⬇️ Descargar PDF", guardado[1], file_name=nombre_pdf, mime="application/pdf",
                           use_container_width=True)
    else:
        b2.button("⬇️ Descargar PDF", disabled=True, use_container_width=True,
                  help="Primero pulse «Generar PDF»" + (" (cambió algún dato)" if guardado else ""))
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        b3.download_button("Descargar gráfico (.png)", buf.getvalue(), file_name="grafico_corte.png",
                           use_container_width=True)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        en_streamlit = get_script_run_ctx() is not None
    except Exception:
        en_streamlit = False
    if en_streamlit:
        app()
    else:
        # Prueba rápida en consola: python app_corte.py
        for nom, ej in EJEMPLOS.items():
            if nom.startswith("✏️"):
                continue
            r = calcular(ej["datos"])
            print(f"{nom}\n   -> {r.distribucion}   ok={r.ok}")
        print("\nPara abrir la app:  streamlit run app_corte.py")
