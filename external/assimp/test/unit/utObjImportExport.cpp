/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2016, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the
following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other
materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
contributors may be used to endorse or promote products
derived from this software without specific prior
written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

#include "UnitTestPCH.h"
#include "SceneDiffer.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

using namespace Assimp;

static const float VertComponents[ 24 * 3 ] = {
    -0.500000,  0.500000,  0.500000,
    -0.500000,  0.500000, -0.500000,
    -0.500000, -0.500000, -0.500000,
    -0.500000, -0.500000,  0.500000,
    -0.500000, -0.500000, -0.500000,
     0.500000, -0.500000, -0.500000,
     0.500000, -0.500000,  0.500000,
    -0.500000, -0.500000,  0.500000,
    -0.500000,  0.500000, -0.500000,
     0.500000,  0.500000, -0.500000,
     0.500000, -0.500000, -0.500000,
    -0.500000, -0.500000, -0.500000,
     0.500000,  0.500000,  0.500000,
     0.500000,  0.500000, -0.500000,
    -0.500000,  0.500000, -0.500000,
    -0.500000,  0.500000,  0.500000,
     0.500000, -0.500000,  0.500000,
     0.500000,  0.500000,  0.500000,
    -0.500000,  0.500000,  0.500000,
    -0.500000, -0.500000,  0.500000,
     0.500000, -0.500000, -0.500000,
     0.500000,  0.500000, -0.500000,
     0.500000,  0.500000,  0.500000f,
     0.500000, -0.500000,  0.500000f
};

static const std::string ObjModel =
    "o 1\n"
    "\n"
    "# Vertex list\n"
    "\n"
    "v -0.5 -0.5  0.5\n"
    "v -0.5 -0.5 -0.5\n"
    "v -0.5  0.5 -0.5\n"
    "v -0.5  0.5  0.5\n"
    "v  0.5 -0.5  0.5\n"
    "v  0.5 -0.5 -0.5\n"
    "v  0.5  0.5 -0.5\n"
    "v  0.5  0.5  0.5\n"
    "\n"
    "# Point / Line / Face list\n"
    "\n"
    "usemtl Default\n"
    "f 4 3 2 1\n"
    "f 2 6 5 1\n"
    "f 3 7 6 2\n"
    "f 8 7 3 4\n"
    "f 5 8 4 1\n"
    "f 6 7 8 5\n"
    "\n"
    "# End of file\n";

class utObjImportExport : public ::testing::Test {
protected:
    virtual void SetUp() {
        m_im = new Assimp::Importer;
    }

    virtual void TearDown() {
        delete m_im;
        m_im = nullptr;
    }

    aiScene *createScene() {
        aiScene *expScene = new aiScene;
        expScene->mNumMeshes = 1;
        expScene->mMeshes = new aiMesh*[ 1 ];
        aiMesh *mesh = new aiMesh;
        mesh->mName.Set( "1" );
        mesh->mNumVertices = 24;
        mesh->mVertices = new aiVector3D[ 24 ];
        ::memcpy( &mesh->mVertices->x, &VertComponents[ 0 ], sizeof( float ) * 24 * 3 );
        mesh->mNumFaces = 6;
        mesh->mFaces = new aiFace[ mesh->mNumFaces ];

        mesh->mFaces[ 0 ].mNumIndices = 4;
        mesh->mFaces[ 0 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 0 ].mIndices[ 0 ] = 0;
        mesh->mFaces[ 0 ].mIndices[ 1 ] = 1;
        mesh->mFaces[ 0 ].mIndices[ 2 ] = 2;
        mesh->mFaces[ 0 ].mIndices[ 3 ] = 3;

        mesh->mFaces[ 1 ].mNumIndices = 4;
        mesh->mFaces[ 1 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 1 ].mIndices[ 0 ] = 4;
        mesh->mFaces[ 1 ].mIndices[ 1 ] = 5;
        mesh->mFaces[ 1 ].mIndices[ 2 ] = 6;
        mesh->mFaces[ 1 ].mIndices[ 3 ] = 7;

        mesh->mFaces[ 2 ].mNumIndices = 4;
        mesh->mFaces[ 2 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 2 ].mIndices[ 0 ] = 8;
        mesh->mFaces[ 2 ].mIndices[ 1 ] = 9;
        mesh->mFaces[ 2 ].mIndices[ 2 ] = 10;
        mesh->mFaces[ 2 ].mIndices[ 3 ] = 11;

        mesh->mFaces[ 3 ].mNumIndices = 4;
        mesh->mFaces[ 3 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 3 ].mIndices[ 0 ] = 12;
        mesh->mFaces[ 3 ].mIndices[ 1 ] = 13;
        mesh->mFaces[ 3 ].mIndices[ 2 ] = 14;
        mesh->mFaces[ 3 ].mIndices[ 3 ] = 15;

        mesh->mFaces[ 4 ].mNumIndices = 4;
        mesh->mFaces[ 4 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 4 ].mIndices[ 0 ] = 16;
        mesh->mFaces[ 4 ].mIndices[ 1 ] = 17;
        mesh->mFaces[ 4 ].mIndices[ 2 ] = 18;
        mesh->mFaces[ 4 ].mIndices[ 3 ] = 19;

        mesh->mFaces[ 5 ].mNumIndices = 4;
        mesh->mFaces[ 5 ].mIndices = new unsigned int[ mesh->mFaces[ 0 ].mNumIndices ];
        mesh->mFaces[ 5 ].mIndices[ 0 ] = 20;
        mesh->mFaces[ 5 ].mIndices[ 1 ] = 21;
        mesh->mFaces[ 5 ].mIndices[ 2 ] = 22;
        mesh->mFaces[ 5 ].mIndices[ 3 ] = 23;

        expScene->mMeshes[ 0 ] = mesh;

        expScene->mNumMaterials = 1;
        expScene->mMaterials = new aiMaterial*[ expScene->mNumMaterials ];

        return expScene;
    }

protected:
    Assimp::Importer *m_im;
    aiScene *m_expectedScene;
};

TEST_F( utObjImportExport, obj_import_test ) {
    const aiScene *scene = m_im->ReadFileFromMemory( (void*) ObjModel.c_str(), ObjModel.size(), 0 );
    aiScene *expected = createScene();
    EXPECT_NE( nullptr, scene );

    SceneDiffer differ;
    EXPECT_TRUE( differ.isEqual( expected, scene ) );
    differ.showReport();
}
