import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const apiUrl = (process.env.API_URL || "https://davidnet-api.onrender.com").replace(/\/+$/, "");
  const { searchParams } = new URL(req.url);
  const explain = searchParams.get("explain") || "false";

  try {
    const formData = await req.formData();
    const res = await fetch(`${apiUrl}/predict?explain=${explain}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.text();
    return new NextResponse(data, {
      status: res.status,
      headers: {
        "content-type": "application/json",
      },
    });
  } catch (err: any) {
    return NextResponse.json(
      { detail: `Edge Gateway Proxy Failed: ${err.message}` },
      { status: 502 }
    );
  }
}
