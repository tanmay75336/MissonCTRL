from app.tools.serpapi import SerpApiTools


def main():

    tools = SerpApiTools()

    results = tools.web_search(
        "student businesses Mumbai"
    )

    print("\n" + "=" * 60)
    print("MISSIONCTRL — SERPAPI TEST")
    print("=" * 60)

    print(
        "Organic results:",
        len(results.get("organic_results", []))
    )

    for result in results.get("organic_results", [])[:5]:

        print("\nTITLE:", result.get("title"))
        print("LINK:", result.get("link"))
        print("SNIPPET:", result.get("snippet"))


if __name__ == "__main__":
    main()