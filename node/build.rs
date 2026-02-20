fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = "../proto";
    let protos = &[
        format!("{}/messages.proto", proto_dir),
        format!("{}/inference.proto", proto_dir),
        format!("{}/training.proto", proto_dir),
    ];

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(protos, &[proto_dir])?;

    for proto in protos {
        println!("cargo:rerun-if-changed={}", proto);
    }

    Ok(())
}
