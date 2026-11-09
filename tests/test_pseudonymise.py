"""Local pseudonymisation (pseudonymise.py): what must be caught, what must survive."""

import pytest

from complaintexp.pseudonymise import Pseudonymiser, Span, _resolve_overlaps


@pytest.mark.parametrize(
    "value, kind",
    [
        ("a@b.com", "EMAIL"),
        ("jan.peeters+shop@example.co.uk", "EMAIL"),
        ("BE68 5390 0754 7034", "IBAN"),
        ("BE68539007547034", "IBAN"),
        ("NL91 ABNA 0417 1643 00", "IBAN"),
        ("FR14 2004 1010 0505 0001 3M02 606", "IBAN"),
        ("85.07.30-033.28", "NATIONAL_ID"),
        ("+32 478 11 22 33", "PHONE"),
        ("0475 12 34 56", "PHONE"),
        ("0475/12.34.56", "PHONE"),
        ("0032 9 123 45 67", "PHONE"),
        ("09 123 45 67", "PHONE"),
        ("+32 9 123 45 67", "PHONE"),
        ("0475123456", "PHONE"),
        ("+33 6 12 34 56 78", "PHONE"),
        ("ORD-4821937", "ORDER"),
        ("ORD4821937", "ORDER"),
        ("CUST-123456", "CUSTOMER_ID"),
    ],
)
def test_structured_identifier_is_redacted(ps, value, kind):
    out, mapping = ps.redact(f"Details: {value}. Thanks", "en")
    assert value not in out
    assert mapping == {f"<{kind}_1>": value}


def test_several_identifiers_in_one_text(ps):
    text = "Mail a@b.com, IBAN BE68 5390 0754 7034, order ORD-4821937, tel +32 478 11 22 33, 85.07.30-033.28"
    out, mapping = ps.redact(text, "en")
    assert len(mapping) == 5
    for original in mapping.values():
        assert original not in out


@pytest.mark.parametrize(
    "text",
    [
        "Delivered on 2026-10-28, amount EUR 89.50.",
        "I paid 129 euro for 2 items on 30/10.",
        "Order total was 1.234,50 and I waited 3 weeks.",
        "Room 12, floor 3.",
        "Bezorgd op 05.11.2026 om 14u.",
        "Livré le 03/11/26, montant 42,00.",
        "Invoice 2026-0042 dated 2026-11-03.",
    ],
)
def test_ordinary_numbers_are_left_alone(ps, text):
    """False positives cost quality (the model loses dates/amounts), so they matter too."""
    out, mapping = ps.redact(text, "en")
    assert out == text and mapping == {}


def test_same_value_gets_same_placeholder_and_restores(ps):
    text = "Write to a@b.com. Again: a@b.com"
    out, mapping = ps.redact(text, "en")
    assert out.count("<EMAIL_1>") == 2 and len(mapping) == 1
    assert Pseudonymiser.restore(out, mapping) == text


def test_same_value_different_case_shares_placeholder(ps):
    out, mapping = ps.redact("A@B.com and a@b.com", "en")
    assert out == "<EMAIL_1> and <EMAIL_1>"


def test_distinct_values_get_numbered_placeholders(ps):
    out, _ = ps.redact("a@b.com, c@d.com", "en")
    assert out == "<EMAIL_1>, <EMAIL_2>"


def test_restore_does_not_confuse_placeholder_1_with_10(ps):
    text = ", ".join(f"u{i}@x.com" for i in range(1, 12))
    out, mapping = ps.redact(text, "en")
    assert "<EMAIL_10>" in out
    assert Pseudonymiser.restore(out, mapping) == text


def test_restore_in_model_reply_with_reused_placeholders():
    reply = "Dear <PERSON_1>, we refunded order <ORDER_1>. Regards"
    mapping = {"<PERSON_1>": "Jan Peeters", "<ORDER_1>": "ORD-4821937"}
    assert Pseudonymiser.restore(reply, mapping) == "Dear Jan Peeters, we refunded order ORD-4821937. Regards"


def test_custom_id_patterns_replace_defaults():
    p = Pseudonymiser(use_ner=False, id_patterns=[("TICKET", r"\bTCK\d{4}\b")])
    out, _ = p.redact("TCK1234 and ORD-4821937", "en")
    assert out == "<TICKET_1> and ORD-4821937"


def test_without_ner_flag_reports_mode(ps):
    assert ps.has_ner is False


def test_overlaps_keep_the_longest_span():
    spans = [Span(0, 5, "A"), Span(2, 12, "B"), Span(12, 14, "C")]
    assert _resolve_overlaps(spans) == [Span(2, 12, "B"), Span(12, 14, "C")]


def test_empty_text(ps):
    assert ps.redact("", "en") == ("", {})


# -- NER mode: only runs when the `redaction` extra and spaCy models are installed
@pytest.mark.ner
@pytest.mark.parametrize(
    "text, lang, name",
    [
        ("Ik ben Jan Peeters en mijn pakket is niet aangekomen.", "nl", "Jan Peeters"),
        ("Bonjour, je suis Marie Dupont et ma commande est en retard.", "fr", "Marie Dupont"),
        ("Hi, this is Sarah Williams about my refund.", "en", "Sarah Williams"),
    ],
)
def test_ner_redacts_person_names(ner_ps, text, lang, name):
    out, _ = ner_ps.redact(text, lang)
    assert name not in out


# -- addresses (regex, so these run with and without NER) ---------------------
@pytest.mark.parametrize(
    "text, address",
    [
        ("Ik ben Jan, Kerkstraat 12, 9000 Gent. Bel me", "Kerkstraat 12, 9000 Gent"),
        ("Adres: Grote Markt 1, 1000 Brussel", "Grote Markt 1, 1000 Brussel"),
        ("Woonachtig Stationsplein 4 bus 2, 2000 Antwerpen.", "Stationsplein 4 bus 2, 2000 Antwerpen"),
        ("Sint-Pietersnieuwstraat 33, 9000 Gent", "Sint-Pietersnieuwstraat 33, 9000 Gent"),
        ("Bezorgd op Leopoldlaan 7A te Brugge", "Leopoldlaan 7A"),
        ("Je habite 12 rue de la Paix, 75002 Paris.", "12 rue de la Paix, 75002 Paris"),
        ("Siège: rue de la Loi 16, 1000 Bruxelles", "rue de la Loi 16, 1000 Bruxelles"),
        ("avenue Louise 54 bte 3, 1050 Ixelles", "avenue Louise 54 bte 3, 1050 Ixelles"),
        ("I live at 221B Baker Street, London NW1 6XE.", "221B Baker Street, London NW1 6XE"),
        ("Send it to 10 Downing St, please", "10 Downing St"),
        ("Damrak 1, 1012 LG Amsterdam", "1012 LG Amsterdam"),
    ],
)
def test_address_is_redacted_as_one_placeholder(ps, text, address):
    out, mapping = ps.redact(text, "en")
    assert mapping == {"<ADDRESS_1>": address}
    assert out == text.replace(address, "<ADDRESS_1>")


@pytest.mark.parametrize(
    "text",
    [
        "In 2025 Amazon was faster.",
        "I paid 1500 EUR for 3 items.",
        "We moeten 3 weken wachten op de markt 2 keer.",
        "Mijn markt 3 dagen",
        "Le magasin a 2 étages et 1 place de parking.",
        "Delivered on 2026-10-28, 3 parcels.",
    ],
)
def test_text_that_is_not_an_address_is_left_alone(ps, text):
    out, mapping = ps.redact(text, "en")
    assert out == text and mapping == {}


def test_year_after_a_comma_can_still_be_a_postcode(ps):
    out, _ = ps.redact("Stationsplein 4, 2000 Antwerpen", "nl")
    assert out == "<ADDRESS_1>"


def test_same_address_twice_shares_placeholder(ps):
    out, mapping = ps.redact("Kerkstraat 12, 9000 Gent. Again: Kerkstraat 12, 9000 Gent", "nl")
    assert out == "<ADDRESS_1>. Again: <ADDRESS_1>" and len(mapping) == 1


def test_address_restores_in_reply(ps):
    text = "Stuur naar Kerkstraat 12, 9000 Gent"
    out, mapping = ps.redact(text, "nl")
    assert Pseudonymiser.restore(out, mapping) == text


def test_lone_city_is_not_merged_into_an_address():
    from complaintexp.pseudonymise import _merge_addresses

    spans = [Span(0, 5, "LOCATION"), Span(10, 20, "PERSON")]
    assert _merge_addresses(spans, "Gent and Jan Peeter") == spans


@pytest.mark.ner
def test_ner_city_next_to_street_becomes_one_address(ner_ps):
    out, mapping = ner_ps.redact("Ik ben Jan Peeters, Kerkstraat 12, 9000 Gent. Bel me", "nl")
    assert "<ADDRESS_1>" in out and mapping["<ADDRESS_1>"] == "Kerkstraat 12, 9000 Gent"


# -- conversations and trailing cities ------------------------------------------
def test_mapping_carries_over_between_messages(ps):
    first, m = ps.redact("Order ORD-1234567, mail a@b.com", "en")
    second, m2 = ps.redact('{"order": "ORD-1234567", "other": "ORD-7654321"}', "en", m)
    assert "<ORDER_1>" in second and "<ORDER_2>" in second
    assert m2["<ORDER_1>"] == "ORD-1234567" and m2["<ORDER_2>"] == "ORD-7654321" and m2["<EMAIL_1>"] == "a@b.com"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Write to 14 Elm Grove, Bristol", "Write to <ADDRESS_1>"),
        ("Kerkstraat 12, Sint-Niklaas. Groeten", "<ADDRESS_1>. Groeten"),
        ("Kerkstraat 12, Dank u", "<ADDRESS_1>, Dank u"),          # not a city: clause goes on
        ("Kerkstraat 12, maar Jan woont elders.", "<ADDRESS_1>, maar Jan woont elders."),
    ],
)
def test_city_right_after_a_street_is_part_of_the_address(ps, text, expected):
    assert ps.redact(text, "nl")[0] == expected
